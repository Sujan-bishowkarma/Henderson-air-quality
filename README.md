# Henderson Air Quality Monitoring and Prediction System

## Project Overview
This joint BBIM713 Capstone Project examines historical PM10 and NO₂ concentrations at Henderson, Auckland, using machine learning, Supabase PostgreSQL, REST API integration and Streamlit.

## Team Members
- **Asmita Khatri:** Team Leader – Decision Tree, LightGBM, Prophet and project coordination.
- **Sujan Bishowkarma:** Linear Regression, Random Forest, XGBoost and technical implementation.

Both team members contributed to preprocessing, exploratory data analysis, cloud integration and dashboard testing.

## Dataset
- **Location:** Henderson, Auckland
- **Period:** January 2006 – February 2023
- **Records:** 150,432 hourly observations
- **Variables:** PM10, NO₂, temperature, relative humidity and wind speed

## Machine Learning Results

| Forecast Target | LightGBM R² |
|---|---:|
| Next-hour NO₂ | 0.762 |
| Next-hour PM10 | 0.333 |
| Next 3-hour average PM10 | 0.441 |
| Next 24-hour average PM10 | 0.387 |

## Technologies
Python, Pandas, Scikit-learn, XGBoost, LightGBM, Prophet, Supabase PostgreSQL and Streamlit.

## Joint Project Portfolio
[View Our GitHub Pages Portfolio](https://sujan-bishowkarma.github.io/Henderson-air-quality/)

## Project Evidence
See the `Screenshots/` folder for exploratory analysis, model evaluation, API testing and dashboard screenshots.

## Project Limitations
This is a historical research prototype and does not provide real-time monitoring or official public-health alerts.
