### Description
Web application prototype for automating pneumonia diagnosis from chest X-ray images. The dataset used contains 5,863 images; it is public and available at https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/

### Deployment
1. Install dependencies: `pip install -r requirements.txt`
2. Download `best_model.pth` from the following link and place it in the `models/` folder: https://upm365-my.sharepoint.com/:u:/g/personal/victor_cadenas_alumnos_upm_es/IQDyG5qZfNwETKkJd44FHfEDAWbWBZdfcLO9hJCph1uum2E?e=0B10yF
3. Run the app: `streamlit run src/app.py` (from the project folder)

### 📂 Project Structure

* **data/**: Sample images (normal/pneumonia).
* **notebooks/**: Training, fine-tuning and evaluation process of the ResNet18 model.
* **logs/**: Detailed per-epoch metrics log (`log-epochs.txt`).
* **models/**: `best_model.pth`, selected using the **F2-score** metric.
* **src/**:
    * `app.py`: Streamlit user interface.
    * `utils.py`: Inference logic and Grad-CAM generation.

### Results (test split)
Accuracy 93.6% · Precision 96.5% · Recall 94.7% · F2-score 95.0%

### LIMITATIONS AND NOTES

1. Although the final model achieves good metrics, it may be biased toward a young, Asian population, since the chest X-rays come from a pediatric unit of a hospital in China.
2. Each time the "Predict" button is pressed, the displayed heatmap is different. This is due to the method used (Smooth Grad-CAM++), which adds random Gaussian noise at each iteration and averages the results to reduce the overall noise of the map. To always generate the same heatmap, a random seed must be fixed.