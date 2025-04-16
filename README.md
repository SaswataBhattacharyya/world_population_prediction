# World Population Prediction

A web application for visualizing and forecasting world population trends using historical data with ARIMA models.

## Features

- **Interactive Data Upload**: Upload your own population dataset in CSV or Excel format
- **Country-Specific Analysis**: Select any country for detailed population analysis
- **Advanced Forecasting**: Generate population forecasts for up to 10 years
- **Comprehensive Visualizations**:
  - Population trends over time
  - Density changes and growth rates
  - Geographic maps showing country locations
  - Population share analysis with pie charts
  
## Technologies Used

- **Backend**: Flask, Python
- **Data Processing**: Pandas, NumPy
- **Statistical Modeling**: StatsModels, pmdarima for ARIMA modeling
- **Visualization**: Plotly, Matplotlib, GeoJSON
- **Geographic Data**: GeoPandas for spatial operations
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/SaswataBhattacharyya/world_population_prediction.git
   cd world_population_prediction
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

## Usage

1. Upload a population dataset that contains the required columns
2. Select a country from the dropdown menu
3. Adjust the start year, forecast years, and pie chart year using the sliders
4. Click "Generate Forecast" to view the results
5. Explore the various visualizations and insights

## Data Requirements

The application expects datasets with the following columns:
- Rank
- CCA3 
- Country/Territory
- Capital
- Continent
- Population columns for years (1970-2022)
- Area (km²)
- Density (per km²)
- Growth Rate
- World Population Percentage

## Project Structure

- `app.py`: Main application file with Flask routes and data processing
- `templates/`: HTML templates
- `static/css/`: CSS stylesheets
- `data/`: Geographic data for map visualization
- `uploads/`: Directory for user-uploaded files
- `saved_models/`: Directory for saved ARIMA models

## License

MIT License

## Author

Saswata Bhattacharyya 