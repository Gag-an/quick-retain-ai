# QuickRetain AI - Setup Guide

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Gag-an/quick-retain-ai.git
cd quick-retain-ai
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📋 System Requirements

### Python Version
- Python 3.8 or higher

### Operating System
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+)

### Memory
- Minimum: 4GB RAM
- Recommended: 8GB+ RAM

## 🔧 Optional Dependencies

For enhanced performance and additional features:

```bash
# Install optional ML libraries
pip install xgboost lightgbm catboost

# Install development tools
pip install pytest black jupyter numba
```

## 📁 Project Structure

```
quick-retain-ai/
├── app.py                 # Main Streamlit application
├── pages/                 # Individual page modules
│   ├── 01_Churn_SHAP.py   # Churn prediction with SHAP
│   ├── 02_Retention_RL.py # Retention reinforcement learning
│   ├── 03_Logistics.py    # Logistics optimization
│   └── 04_Campaigns.py    # Campaign management
├── data/                  # Data directories
│   ├── raw/              # Raw data files
│   ├── cleaned/          # Processed data files
│   └── processed/        # ML pipeline outputs
├── models/               # Trained ML models
├── scripts/              # Training and utility scripts
├── clean_datasets.py     # Data processing utilities
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies
└── README.md            # Project documentation
```

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: No module named 'matplotlib'**
   ```bash
   pip install matplotlib
   ```

2. **ImportError: No module named 'sklearn'**
   ```bash
   pip install scikit-learn
   ```

3. **ImportError: No module named 'shap'**
   ```bash
   pip install shap
   ```

4. **Memory issues with large datasets**
   - Reduce sample size in the app
   - Use a machine with more RAM
   - Consider using data sampling

### Platform-Specific Issues

#### Windows
- Ensure you have Visual C++ Build Tools installed
- Use Anaconda if pip installation fails

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies

#### Linux
- Install system dependencies: `sudo apt-get install python3-tk libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev`

## 📊 Data Requirements

### Input Data Format
- CSV files in `data/cleaned/` directory
- Required columns vary by module (see individual page documentation)
- Data should be preprocessed and cleaned

### Sample Data
The application includes synthetic data generation for testing when no real data is available.

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. The `requirements.txt` and `packages.txt` files will be used automatically

### Local Production
1. Install dependencies: `pip install -r requirements.txt`
2. Run with: `streamlit run app.py --server.port 8501`
3. Access at: `http://localhost:8501`

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the individual page documentation
3. Check the GitHub issues page
4. Create a new issue with detailed error information

## 🔄 Updates

To update the application:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**QuickRetain AI** - Smart retention made simple! 🎯
