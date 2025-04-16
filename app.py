from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.utils
import json
import os
import traceback
from datetime import datetime
import joblib
import warnings
from werkzeug.utils import secure_filename
from pmdarima import auto_arima
import geopandas as gpd
import plotly.express as px

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MODEL_FOLDER'] = 'saved_models/arima'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global variables
uploaded_data = None

def validate_file_format(df):
    """Validate the uploaded file has the necessary columns"""
    required_columns = [
        'Rank', 'CCA3', 'Country/Territory', 'Capital', 'Continent',
        '2022 Population', '2020 Population', '2015 Population',
        '2010 Population', '2000 Population', '1990 Population',
        '1980 Population', '1970 Population', 'Area (km²)',
        'Density (per km²)', 'Growth Rate', 'World Population Percentage'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    return True, "File format is valid"

def process_data(df):
    """Process data exactly as in feature_model_final.ipynb"""
    # Melt the Data
    pop_cols = [col for col in df.columns if 'Population' in col and 'World' not in col]
    df_melted = df.melt(
        id_vars=['Country/Territory', 'Area (km²)', 'Growth Rate'],
        value_vars=pop_cols,
        var_name='Year',
        value_name='Population'
    )
    df_melted['Year'] = df_melted['Year'].str.extract(r'(\d{4})').astype(int)
    df_melted['Date'] = pd.to_datetime(df_melted['Year'].astype(str) + '-01-01')
    df_melted.sort_values(by=['Country/Territory', 'Date'], inplace=True)
    
    # Create all years dataframe and interpolate missing years
    all_years = pd.DataFrame({'Year': range(1970, 2023)})
    interpolated_dfs = []
    
    for country in df_melted['Country/Territory'].unique():
        country_df = df_melted[df_melted['Country/Territory'] == country].copy()
        
        # Merge with all_years to get all years from 1970 to 2022
        merged = pd.merge(all_years, country_df, on='Year', how='left')
        merged['Country/Territory'] = country
        merged['Area (km²)'] = merged['Area (km²)'].ffill()
        
        # Interpolate population for missing years
        merged['Population'] = merged['Population'].interpolate(method='linear')
        
        # Recompute Date and Density
        merged['Date'] = pd.to_datetime(merged['Year'].astype(str) + '-01-01')
        merged['Density'] = merged['Population'] / merged['Area (km²)']
        
        # Calculate growth rates as percent change
        merged['Growth'] = merged['Population'].pct_change()
        
        # Set first year's growth to match second year (extrapolate first year)
        if len(merged) > 1:
            merged.loc[merged.index[0], 'Growth'] = merged.loc[merged.index[1], 'Growth']
        else:
            merged['Growth'] = 0  # Handle case where only one year exists
        
        interpolated_dfs.append(merged)
    
    # Combine all countries into one dataframe
    df_full = pd.concat(interpolated_dfs, ignore_index=True)
    df_full.sort_values(by=['Country/Territory', 'Year'], inplace=True)
    
    return df_full

def generate_forecasts(df_full, forecast_years):
    """Generate forecasts for all countries using saved ARIMA models or creating new ones"""
    forecast_results = {}
    
    # Process each country
    for country in df_full['Country/Territory'].unique():
        country_df = df_full[df_full['Country/Territory'] == country].sort_values('Year')
        
        # Skip if not enough data
        if len(country_df) < 2:
            app.logger.warning(f"Skipping {country}: Not enough data points")
            continue
            
        # Get growth series
        growth_series = country_df['Growth'].dropna().astype(float)
        
        # Model file path
        model_filename = f"arima_model_{country.replace('/', '_')}.pkl"
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
        app.logger.info(f"Looking for model at: {model_path}")
        
        # Try to load existing model, create new one if needed
        try:
            if os.path.exists(model_path):
                # Load existing model
                app.logger.info(f"Loading existing model for {country}")
                try:
                    growth_model = joblib.load(model_path)
                    app.logger.info(f"Model loaded successfully for {country}")
                except Exception as load_error:
                    app.logger.error(f"Error loading model for {country}: {str(load_error)}")
                    app.logger.error(f"Creating new model for {country}")
                    growth_model = auto_arima(growth_series, seasonal=False, suppress_warnings=True)
                    
                    # Save model
                    try:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        joblib.dump(growth_model, model_path)
                        app.logger.info(f"New model saved for {country}")
                    except Exception as save_error:
                        app.logger.error(f"Error saving model for {country}: {str(save_error)}")
            else:
                # Create new model
                app.logger.info(f"No existing model found. Creating new model for {country}")
                growth_model = auto_arima(growth_series, seasonal=False, suppress_warnings=True)
                
                # Save model
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    joblib.dump(growth_model, model_path)
                    app.logger.info(f"New model saved for {country}")
                except Exception as save_error:
                    app.logger.error(f"Error saving model for {country}: {str(save_error)}")
            
            # Generate forecast
            app.logger.info(f"Generating forecast for {country}")
            growth_forecast = growth_model.predict(n_periods=forecast_years)
            
            # Ensure forecast is a numpy array
            if hasattr(growth_forecast, 'values'):
                growth_forecast = growth_forecast.values
            
            # Store results
            forecast_results[country] = {
                'growth_forecast': growth_forecast
            }
            app.logger.info(f"Forecast for {country} completed successfully")
        except Exception as e:
            app.logger.error(f"Error forecasting for {country}: {str(e)}")
            app.logger.error(traceback.format_exc())
            continue
    
    return forecast_results

def create_forecast_dataframe(df_full, forecast_results, forecast_years):
    """Create full dataframe with forecasted data"""
    # Create copy of original dataframe for adding forecasts
    df_with_forecasts = df_full.copy()
    
    # Add column to mark historical vs forecasted data
    df_with_forecasts['is_forecast'] = False
    
    # Create dataframe for forecasted data
    forecast_data = []
    
    # Process each country
    for country, forecasts in forecast_results.items():
        # Get country data
        country_data = df_full[df_full['Country/Territory'] == country].sort_values('Year')
        
        if len(country_data) == 0:
            continue
            
        # Get last year's data
        last_year = country_data['Year'].max()
        last_population = country_data[country_data['Year'] == last_year]['Population'].values[0]
        area = country_data['Area (km²)'].values[0]
        growth_forecast = forecasts['growth_forecast']
        
        # Convert to numpy array if it's a pandas Series to ensure we can access by integer index
        if hasattr(growth_forecast, 'values'):
            growth_forecast = growth_forecast.values
        
        # Calculate forecasted populations
        population = last_population
        for i in range(forecast_years):
            try:
                # Get next year
                year = last_year + i + 1
                
                # Apply growth rate to get new population
                population = population * (1 + growth_forecast[i])
                
                # Calculate density
                density = population / area
                
                # Add to forecast data
                forecast_data.append({
                    'Country/Territory': country,
                    'Year': year,
                    'Population': population,
                    'Density': density,
                    'Growth': growth_forecast[i],
                    'Area (km²)': area,
                    'Date': pd.to_datetime(f"{year}-01-01"),
                    'is_forecast': True
                })
            except Exception as e:
                app.logger.error(f"Error processing forecast for {country} at index {i}: {str(e)}")
                app.logger.error(f"Growth forecast type: {type(growth_forecast)}")
                app.logger.error(f"Growth forecast: {growth_forecast}")
                break
    
    # Convert to dataframe
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        
        # Combine with original data
        df_with_forecasts = pd.concat([df_with_forecasts, forecast_df], ignore_index=True)
    
    return df_with_forecasts

def create_plots(country, start_year, pie_chart_year, df_full, df_with_forecasts):
    """Create plots for the selected country"""
    # Filter data for the country and starting year
    country_data = df_with_forecasts[
        (df_with_forecasts['Country/Territory'] == country) & 
        (df_with_forecasts['Year'] >= start_year)
    ].copy()
    
    if country_data.empty:
        raise ValueError(f"No data available for {country} from year {start_year}")
    
    # Split into historical and forecast data
    historical_data = country_data[country_data['is_forecast'] == False]
    forecast_data = country_data[country_data['is_forecast'] == True]
    
    # Create population plot
    pop_fig = go.Figure()
    
    # Historical data
    pop_fig.add_trace(go.Scatter(
        x=historical_data['Year'],
        y=historical_data['Population'],
        name='Historical Population',
        line=dict(color='blue')
    ))
    
    # Forecast data
    if not forecast_data.empty:
        pop_fig.add_trace(go.Scatter(
            x=forecast_data['Year'],
            y=forecast_data['Population'],
            name='Forecasted Population',
            line=dict(color='red', dash='dash')
        ))
    
    pop_fig.update_layout(
        title=f'Population Trend for {country}',
        xaxis_title='Year',
        yaxis_title='Population'
    )
    
    # Create density plot
    density_fig = go.Figure()
    
    # Historical density
    density_fig.add_trace(go.Scatter(
        x=historical_data['Year'],
        y=historical_data['Density'],
        name='Historical Density',
        line=dict(color='green')
    ))
    
    # Forecast density
    if not forecast_data.empty:
        density_fig.add_trace(go.Scatter(
            x=forecast_data['Year'],
            y=forecast_data['Density'],
            name='Forecasted Density',
            line=dict(color='orange', dash='dash')
        ))
    
    density_fig.update_layout(
        title=f'Population Density Trend for {country}',
        xaxis_title='Year',
        yaxis_title='Density (per km²)'
    )
    
    # Create growth rate plot
    growth_fig = go.Figure()
    
    # Historical growth rate
    growth_fig.add_trace(go.Scatter(
        x=historical_data['Year'],
        y=historical_data['Growth'].fillna(0),
        name='Historical Growth Rate',
        line=dict(color='purple')
    ))
    
    # Forecast growth rate
    if not forecast_data.empty:
        growth_fig.add_trace(go.Scatter(
            x=forecast_data['Year'],
            y=forecast_data['Growth'],
            name='Forecasted Growth Rate',
            line=dict(color='magenta', dash='dash')
        ))
    
    growth_fig.update_layout(
        title=f'Population Growth Rate for {country}',
        xaxis_title='Year',
        yaxis_title='Growth Rate'
    )
    
    # Create world population plot
    world_pop_fig = go.Figure()
    
    # Get world historical data (sum of all countries per year)
    world_historical = df_full[df_full['Year'] >= start_year].groupby('Year')['Population'].sum().reset_index()
    
    # Add historical world population
    world_pop_fig.add_trace(go.Scatter(
        x=world_historical['Year'],
        y=world_historical['Population'],
        name='World Historical Population',
        line=dict(color='#4e73df', width=3)
    ))
    
    # If we have forecast data, add world forecast
    if not forecast_data.empty:
        max_forecast_year = forecast_data['Year'].max()
        min_forecast_year = forecast_data['Year'].min()
        
        # Get world forecast data (sum of all countries per forecasted year)
        world_forecast = df_with_forecasts[
            (df_with_forecasts['is_forecast'] == True) & 
            (df_with_forecasts['Year'] >= min_forecast_year) &
            (df_with_forecasts['Year'] <= max_forecast_year)
        ].groupby('Year')['Population'].sum().reset_index()
        
        # Add forecasted world population 
        world_pop_fig.add_trace(go.Scatter(
            x=world_forecast['Year'],
            y=world_forecast['Population'],
            name='World Forecasted Population',
            line=dict(color='#1cc88a', width=3, dash='dash')
        ))
    
    # Add country data to the world plot (on secondary y-axis)
    world_pop_fig.add_trace(go.Scatter(
        x=historical_data['Year'],
        y=historical_data['Population'],
        name=f'{country} Historical',
        line=dict(color='#f6c23e', width=2),
        yaxis='y2'
    ))
    
    if not forecast_data.empty:
        world_pop_fig.add_trace(go.Scatter(
            x=forecast_data['Year'],
            y=forecast_data['Population'],
            name=f'{country} Forecasted',
            line=dict(color='#e74a3b', width=2, dash='dash'),
            yaxis='y2'
        ))
    
    world_pop_fig.update_layout(
        title=f'World Population vs {country}',
        xaxis_title='Year',
        yaxis=dict(
            title='World Population',
            titlefont=dict(color='#4e73df'),
            tickfont=dict(color='#4e73df')
        ),
        yaxis2=dict(
            title=f'{country} Population',
            titlefont=dict(color='#f6c23e'),
            tickfont=dict(color='#f6c23e'),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            y=-0.2
        )
    )
    
    # Create historical pie chart using the specified year directly from df_full
    # Filter the data to get only the selected year
    historical_year = int(pie_chart_year)
    year_data = df_full[df_full['Year'] == historical_year].copy()
    
    if len(year_data) > 0:
        world_population = year_data['Population'].sum()
        country_population = year_data[year_data['Country/Territory'] == country]['Population'].values[0] if country in year_data['Country/Territory'].values else 0
        
        # Calculate percentage
        if world_population > 0 and country_population > 0:
            country_percentage = (country_population / world_population) * 100
            app.logger.info(f"Historical pie chart ({historical_year}): Country: {country}, Population: {country_population}, World: {world_population}")
            app.logger.info(f"Historical percentage: {country_percentage}%")
            
            current_pie = go.Figure(data=[go.Pie(
                labels=[country, 'Rest of World'],
                values=[country_population, world_population - country_population],
                hole=0.4,
                marker=dict(
                    colors=['#f6c23e', '#4e73df']
                )
            )])
            
            current_pie.update_layout(
                title={
                    'text': f'{country} Share ({historical_year}): {country_percentage:.6f}%',
                    'y': 0.9,
                    'yanchor': 'top'
                }
            )
        else:
            current_pie = go.Figure(data=[go.Pie(
                labels=['No Data'],
                values=[1],
                hole=0.4,
                marker=dict(colors=['#e0e0e0'])
            )])
            
            current_pie.update_layout(
                title={
                    'text': f'No data for {country} in {historical_year}',
                    'y': 0.9,
                    'yanchor': 'top'
                }
            )
    else:
        current_pie = go.Figure(data=[go.Pie(
            labels=['No Data'],
            values=[1],
            hole=0.4,
            marker=dict(colors=['#e0e0e0'])
        )])
        
        current_pie.update_layout(
            title={
                'text': f'No data available for {historical_year}',
                'y': 0.9,
                'yanchor': 'top'
            }
        )
    
    # Create future pie chart from forecasted data
    if not forecast_data.empty:
        future_year = forecast_data['Year'].max()
        future_year_data = df_with_forecasts[df_with_forecasts['Year'] == future_year]
        
        if not future_year_data.empty:
            future_world_population = future_year_data['Population'].sum()
            future_country_population = future_year_data[future_year_data['Country/Territory'] == country]['Population'].values[0]
            
            # Calculate future percentage
            future_percentage = (future_country_population / future_world_population) * 100
            app.logger.info(f"Future pie chart ({future_year}): Country: {country}, Population: {future_country_population}, World: {future_world_population}")
            app.logger.info(f"Future percentage: {future_percentage}%")
            
            future_pie = go.Figure(data=[go.Pie(
                labels=[country, 'Rest of World'],
                values=[future_country_population, future_world_population - future_country_population],
                hole=0.4,
                marker=dict(
                    colors=['#e74a3b', '#1cc88a']
                )
            )])
            
            future_pie.update_layout(
                title={
                    'text': f'Projected {country} Share ({future_year}): {future_percentage:.6f}%',
                    'y': 0.9,
                    'yanchor': 'top'
                }
            )
        else:
            future_pie = go.Figure(data=[go.Pie(
                labels=['No Data'],
                values=[1],
                hole=0.4,
                marker=dict(colors=['#e0e0e0'])
            )])
            
            future_pie.update_layout(
                title={
                    'text': f'No forecast data for {future_year}',
                    'y': 0.9,
                    'yanchor': 'top'
                }
            )
    else:
        future_pie = go.Figure(data=[go.Pie(
            labels=['No Data'],
            values=[1],
            hole=0.4,
            marker=dict(colors=['#e0e0e0'])
        )])
        
        future_pie.update_layout(
            title={
                'text': 'No forecast data available',
                'y': 0.9,
                'yanchor': 'top'
            }
        )
    
    # Create plots
    plots = {
        'world_population': json.loads(json.dumps(world_pop_fig, cls=plotly.utils.PlotlyJSONEncoder)),
        'population': json.loads(json.dumps(pop_fig, cls=plotly.utils.PlotlyJSONEncoder)),
        'density': json.loads(json.dumps(density_fig, cls=plotly.utils.PlotlyJSONEncoder)),
        'growth': json.loads(json.dumps(growth_fig, cls=plotly.utils.PlotlyJSONEncoder)),
        'current_pie': json.loads(json.dumps(current_pie, cls=plotly.utils.PlotlyJSONEncoder)),
        'future_pie': json.loads(json.dumps(future_pie, cls=plotly.utils.PlotlyJSONEncoder)),
        'country_map': create_country_map(country, df_full)
    }
    
    return plots

def create_country_map(country, df_full):
    """Create a map highlighting the selected country"""
    try:
        # Load the shapefile
        shapefile_path = 'data/10m_cultural/10m_cultural/ne_10m_admin_0_countries.shp'
        print(f"Loading shapefile from: {shapefile_path}")
        print(f"Current directory: {os.getcwd()}")
        
        if not os.path.exists(shapefile_path):
            print(f"Shapefile not found at: {shapefile_path}")
            # Try to find the file
            for root, dirs, files in os.walk('data'):
                for file in files:
                    if file.endswith('.shp') and 'countries' in file:
                        potential_path = os.path.join(root, file)
                        print(f"Found potential shapefile: {potential_path}")
                        if 'ne_10m_admin_0_countries.shp' in potential_path:
                            shapefile_path = potential_path
                            print(f"Using shapefile: {shapefile_path}")
                            break
        
        world = gpd.read_file(shapefile_path)
        
        # Standardize country names
        # Create a mapping between common country names and their NAME field in the shapefile
        country_name_mapping = {
            'United States': 'United States of America',
            'USA': 'United States of America',
            'UK': 'United Kingdom',
            'Russia': 'Russian Federation',
            'South Korea': 'Korea, Republic of',
            'North Korea': "Korea, Dem. People's Rep.",
            'Iran': 'Iran, Islamic Republic of',
            'Syria': 'Syrian Arab Republic',
            'Vietnam': 'Viet Nam',
            'Venezuela': 'Venezuela, Bolivarian Republic of',
            'Bolivia': 'Bolivia, Plurinational State of',
            'Tanzania': 'Tanzania, United Republic of',
            'Congo': 'Congo, Democratic Republic of the'
        }
        
        # Try to find the country in the shapefile
        map_country = country_name_mapping.get(country, country)
        
        # Create a column for coloring
        world['color'] = 'lightblue'
        
        # Find the row for the selected country
        matching_rows = world[world['NAME'].str.contains(map_country, case=False, na=False)]
        
        if not matching_rows.empty:
            # If found, set the color for the selected country
            world.loc[matching_rows.index, 'color'] = 'green'
        else:
            # If not found, try with another column
            matching_rows = world[world['ADMIN'].str.contains(map_country, case=False, na=False)]
            if not matching_rows.empty:
                world.loc[matching_rows.index, 'color'] = 'green'
            else:
                print(f"Country {country} not found in the shapefile")
        
        # Create a choropleth map with Plotly Express
        fig = px.choropleth(
            world,
            geojson=world.geometry,
            locations=world.index,
            color='color',
            color_discrete_map={'lightblue': 'lightblue', 'green': 'green'},
            projection='natural earth',
            title=f'World Map with {country} Highlighted'
        )
        
        # Customize the map
        fig.update_geos(
            showcoastlines=True, coastlinecolor="Black",
            showland=True, landcolor="White",
            showocean=True, oceancolor="LightBlue",
            showlakes=True, lakecolor="LightBlue",
            showcountries=True, countrycolor="Black",
            fitbounds="locations"
        )
        
        # Adjust layout
        fig.update_layout(
            width=1500,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            title={
                'text': f'World Map with {country} Highlighted',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            }
        )
        
        return fig.to_json()
    
    except Exception as e:
        print(f"Error creating map: {e}")
        # Create a simple figure as fallback
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not create map: {str(e)}",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            width=1500, 
            height=600,
            title={
                'text': 'World Map (Error Loading)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            }
        )
        return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_data
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        return jsonify({'error': 'Invalid file format'})
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            uploaded_data = pd.read_csv(file)
        else:
            uploaded_data = pd.read_excel(file)
        
        # Validate format
        is_valid, message = validate_file_format(uploaded_data)
        if not is_valid:
            uploaded_data = None
            return jsonify({'error': message})
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process to get df_full
        df_full = process_data(uploaded_data)
        
        # Get countries and years
        countries = sorted(df_full['Country/Territory'].unique())
        min_year = int(df_full['Year'].min())
        max_year = int(df_full['Year'].max())
        
        return jsonify({
            'success': True,
            'countries': countries,
            'min_year': min_year,
            'max_year': max_year
        })
    
    except Exception as e:
        app.logger.error(f"Error in upload_file: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/forecast', methods=['POST'])
def forecast():
    global uploaded_data
    
    if uploaded_data is None:
        return jsonify({'error': 'Please upload a dataset first'})
    
    try:
        # Get parameters
        country = request.form.get('country')
        if not country:
            return jsonify({'error': 'Country parameter is required'})
            
        try:
            start_year = int(request.form.get('start_year'))
        except:
            return jsonify({'error': 'Invalid start year'})
            
        try:
            forecast_years = int(request.form.get('forecast_years'))
            if forecast_years < 1 or forecast_years > 30:
                return jsonify({'error': 'Forecast years must be between 1 and 30'})
        except:
            return jsonify({'error': 'Invalid forecast years'})
        
        try:
            pie_chart_year = int(request.form.get('pie_chart_year'))
        except:
            return jsonify({'error': 'Invalid pie chart year'})
        
        app.logger.info(f"Starting forecast for country: {country}, start_year: {start_year}, forecast_years: {forecast_years}, pie_chart_year: {pie_chart_year}")
        
        # Process data to get df_full
        df_full = process_data(uploaded_data)
        app.logger.info(f"Data processed successfully. Shape: {df_full.shape}")
        
        # Check if country exists in the data
        available_countries = df_full['Country/Territory'].unique()
        if country not in available_countries:
            return jsonify({'error': f"Country '{country}' not found in the dataset"})
        
        # Generate forecasts
        try:
            forecast_results = generate_forecasts(df_full, forecast_years)
            app.logger.info(f"Forecasts generated for {len(forecast_results)} countries")
            
            if country not in forecast_results:
                app.logger.error(f"No forecast generated for {country}")
                return jsonify({'error': f"Failed to generate forecast for {country}. Check the logs for details."})
        except Exception as e:
            app.logger.error(f"Error in generate_forecasts: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f"Failed to generate forecast: {str(e)}"})
        
        # Create complete dataframe with forecasts
        try:
            df_with_forecasts = create_forecast_dataframe(df_full, forecast_results, forecast_years)
            app.logger.info(f"Forecast dataframe created successfully. Shape: {df_with_forecasts.shape}")
        except Exception as e:
            app.logger.error(f"Error in create_forecast_dataframe: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f"Failed to create forecast dataframe: {str(e)}"})
        
        # Create plots
        try:
            plots = create_plots(country, start_year, pie_chart_year, df_full, df_with_forecasts)
            app.logger.info(f"Plots created successfully")
            
            # Ensure we have a country map
            if 'country_map' not in plots:
                plots['country_map'] = create_country_map(country, df_full)
                app.logger.info(f"Added country map separately")
        except Exception as e:
            app.logger.error(f"Error in create_plots: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f"Failed to create plots: {str(e)}"})
        
        return jsonify(plots)
    
    except Exception as e:
        app.logger.error(f"Error in forecast: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error generating forecast: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5001) 