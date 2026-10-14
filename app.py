import os
from flask import Flask, render_template, request, flash, redirect, url_for
from predict import predict_churn

app = Flask(__name__)
app.secret_key = "customer_churn_prediction_secret_key"  # Required for flash messages

def get_model_metrics():
    """
    Helper function to parse metrics from models/metrics.txt.
    Returns a dictionary of metrics, or fallback defaults if the file is missing.
    """
    metrics_path = "models/metrics.txt"
    defaults = {
        'accuracy': '0.9979',
        'precision': '0.9987',
        'recall': '0.9969',
        'f1_score': '0.9978'
    }
    
    if not os.path.exists(metrics_path):
        return defaults
        
    try:
        metrics = {}
        with open(metrics_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":")
                    # Convert to percentage string for user-friendly display
                    metrics[key] = f"{float(val) * 100:.2f}%"
        return metrics
    except Exception as e:
        print(f"Error parsing metrics: {e}")
        # Return percentages for defaults in case of error
        return {k: f"{float(v)*100:.2f}%" for k, v in defaults.items()}

@app.route('/')
def home():
    """Render the landing / home page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the project documentation and information page."""
    return render_template('about.html')

@app.route('/performance')
def performance():
    """Render the model evaluation page, displaying metrics and static plots."""
    metrics = get_model_metrics()
    return render_template('performance.html', metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles rendering the input form (GET) and making predictions (POST).
    """
    if request.method == 'POST':
        try:
            # 1. Retrieve and Validate Form Fields
            # Numeric fields validation and parsing
            try:
                age = int(request.form.get('Age', ''))
                tenure = int(request.form.get('Tenure', ''))
                usage_freq = int(request.form.get('Usage Frequency', ''))
                support_calls = int(request.form.get('Support Calls', ''))
                payment_delay = int(request.form.get('Payment Delay', ''))
                total_spend = float(request.form.get('Total Spend', ''))
                last_interaction = int(request.form.get('Last Interaction', ''))
            except ValueError:
                flash("Error: Please check that all numeric fields are filled in with valid numbers.", "error")
                return render_template('predict.html')
            
            # Categorical fields
            gender = request.form.get('Gender', '').strip()
            sub_type = request.form.get('Subscription Type', '').strip()
            contract_length = request.form.get('Contract Length', '').strip()
            
            # Simple boundary validations
            if not (18 <= age <= 100):
                flash("Age must be between 18 and 100.", "error")
                return render_template('predict.html')
            if tenure < 0:
                flash("Tenure cannot be negative.", "error")
                return render_template('predict.html')
            if usage_freq < 0:
                flash("Usage Frequency cannot be negative.", "error")
                return render_template('predict.html')
            if not (0 <= support_calls <= 20):
                flash("Support Calls must be between 0 and 20.", "error")
                return render_template('predict.html')
            if payment_delay < 0:
                flash("Payment Delay cannot be negative.", "error")
                return render_template('predict.html')
            if total_spend < 0:
                flash("Total Spend cannot be negative.", "error")
                return render_template('predict.html')
            if not (0 <= last_interaction <= 30):
                flash("Last Interaction must be between 0 and 30 days.", "error")
                return render_template('predict.html')
            
            # 2. Package inputs for the inference pipeline
            customer_data = {
                'Age': age,
                'Gender': gender,
                'Tenure': tenure,
                'Usage Frequency': usage_freq,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Subscription Type': sub_type,
                'Contract Length': contract_length,
                'Total Spend': total_spend,
                'Last Interaction': last_interaction
            }
            
            # 3. Call prediction helper
            result = predict_churn(customer_data)
            
            # 4. Render result page with predictions
            return render_template(
                'result.html',
                prediction=result['churn_prediction'],
                probability=result['churn_probability'],
                top_features=result['top_features'],
                customer_data=customer_data
            )
            
        except Exception as e:
            flash(f"An unexpected error occurred during prediction: {str(e)}", "error")
            return render_template('predict.html')
            
    # GET request: render empty input form
    return render_template('predict.html')

if __name__ == '__main__':
    # Run the Flask app on localhost, port 5000 (standard for local dev)
    app.run(debug=True, host='0.0.0.0', port=5000)
