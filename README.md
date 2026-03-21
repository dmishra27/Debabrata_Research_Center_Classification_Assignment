# Research Centre Quality Classification

## Setup

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\\Scripts\\activate       # Windows

pip install -r requirements.txt
```

## Run

1. Execute `EDA_and_Model.ipynb` to train and save `model_bundle.pkl`.
2. Start the API:

```bash
uvicorn app:app --reload
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{
           "internalFacilitiesCount": 9,
           "hospitals_10km": 3,
           "pharmacies_10km": 2,
           "facilityDiversity_10km": 0.82,
           "facilityDensity_10km": 0.45
         }'
```

**Response:**
```json
{
  "predictedCategory": "Premium",
  "cluster": 1,
  "confidence_note": "Prediction based on K-Means clustering trained on 50 UK research centres."
}
```

## File Structure

```
research-center-assignment/
├── research_centers.csv
├── EDA_and_Model.ipynb
├── app.py
├── requirements.txt
└── README.md
```
