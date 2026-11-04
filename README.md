### Description
Web application prototype for automating pneumonia diagnosis from chest X-ray images. The dataset used contains 5,863 images; it is public and available at https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/

### Results (test split)
Accuracy 93.6% · Precision 96.5% · Recall 94.7% · F2-score 95.0%

### Deployment
The trained model (`models/best_model.pth`, ~45 MB) is stored with [Git LFS](https://git-lfs.com/), so it is downloaded automatically when you clone the repository.

1. Install Git LFS (once per machine): `git lfs install`
2. Clone the repository: `git clone https://github.com/victor-cadenas/REPO-NAME.git` and enter the project folder. If you cloned it before installing Git LFS, run `git lfs pull`.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run src/app.py` (from the project folder)

> **Note:** GitHub's "Download ZIP" button does not include Git LFS files, so the model would be missing. Use `git clone` instead.

### 📂 Project Structure

* **data/**: Sample images (normal/pneumonia).
* **notebooks/**: Training, fine-tuning and evaluation process of the ResNet18 model.
* **logs/**: Detailed per-epoch metrics log (`log-epochs.txt`).
* **models/**: `best_model.pth` (stored with Git LFS), selected using the **F2-score** metric.
* **src/**:
    * `app.py`: Streamlit user interface.
    * `utils.py`: Inference logic and Grad-CAM generation.

### LIMITATIONS AND NOTES

1. Although the final model achieves good metrics, it may be biased toward a young, Asian population, since the chest X-rays come from a pediatric unit of a hospital in China.
2. Each time the "Predict" button is pressed, the displayed heatmap is different. This is due to the method used (Smooth Grad-CAM++), which adds random Gaussian noise at each iteration and averages the results to reduce the overall noise of the map. To always generate the same heatmap, a random seed must be fixed.