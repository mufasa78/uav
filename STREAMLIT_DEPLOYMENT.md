# Deploying to Streamlit Cloud

This guide explains how to deploy the UAV Path Planning Simulation application on Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. A Streamlit Cloud account (sign up at [https://streamlit.io/cloud](https://streamlit.io/cloud))

## Deployment Steps

### 1. Prepare Your Repository

1. Push your code to a GitHub repository.
2. Make sure the following files are included in your repository:
   - `streamlit_app.py`: The main Streamlit application file
   - `.streamlit/config.toml`: Configuration file for Streamlit
   - `streamlit_requirements.txt`: List of Python packages required for the app

### 2. Deploy on Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud).
2. Click on "New app" button.
3. Connect your GitHub account if you haven't already.
4. Select the repository containing your Streamlit app.
5. Configure the deployment:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.9 or later
   - **Requirements file**: `streamlit_requirements.txt`
   - **Advanced settings**: You can configure environment variables if needed

6. Click "Deploy" button.

### 3. Customizing Your App

You can customize your Streamlit app by modifying the `.streamlit/config.toml` file. This file allows you to change:
- Theme colors
- Font settings
- Default page layout
- And more

### 4. Updating Your App

Any changes pushed to the connected GitHub repository will automatically trigger a redeployment of your Streamlit app.

### 5. Viewing Logs and Monitoring

Streamlit Cloud provides logs and usage metrics for your deployed application. You can access these from the app's dashboard.

## Important Files

- `streamlit_app.py`: Main application file
- `.streamlit/config.toml`: Streamlit configuration file
- `streamlit_requirements.txt`: Python package dependencies

## Local Testing

Before deploying, you can test your app locally by running:

```bash
./run_streamlit.sh
```

This will start the Streamlit server on port 8501.

## Troubleshooting

- **Missing Dependencies**: Make sure all required packages are listed in `streamlit_requirements.txt`
- **Import Errors**: Check for correct import paths
- **File Not Found Errors**: Ensure all referenced files exist in the repository
- **Memory Limits**: Streamlit Cloud has memory limits, so optimize your application accordingly

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io/)