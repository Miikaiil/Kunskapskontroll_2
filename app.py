import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# Test-data (om du inte laddat in MNIST än)
test_image = np.random.rand(28, 28)

print("Testar HOG-extrahering...")
try:
    features, hog_image = hog(test_image, orientations=9, 
                              pixels_per_cell=(8, 8), 
                              cells_per_block=(2, 2), 
                              visualize=True)
    print("Succé! HOG fungerar utan blockering.")
except Exception as e:
    print(f"Fel uppstod: {e}")