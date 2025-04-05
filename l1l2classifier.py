import os
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nilearn import plotting
import glob
from nilearn import image
import warnings
warnings.filterwarnings("ignore")

# Step 1: Set up paths and parameters
# Replace this with your dataset root path
data_root = 'adultlanglearn_dataset/'
derivatives_dir = os.path.join(data_root, 'derivatives')
mask_file = os.path.join(derivatives_dir, 'mask', 'language.img')

# Step 2: Collect all relevant functional files
def collect_functional_files(data_root):
    """Collect all functional files and organize by L1/L2"""
    l1_files = []  # Will store paths to L1 files
    l2_files = []  # Will store paths to L2 files
    
    # Loop through all subject folders
    for subj_id in range(1, 35):  # Subjects 01-34
        subj_folder = os.path.join(data_root, f'sub-{subj_id:02d}')
        func_folder = os.path.join(subj_folder, 'func')
        
        if not os.path.exists(func_folder):
            print(f"Warning: No func folder for subject {subj_id}")
            continue
        
        # Look for all functional files
        for filename in os.listdir(func_folder):
            if filename.endswith('_bold.nii.gz'):  # Only get the BOLD files
                filepath = os.path.join(func_folder, filename)
                
                # Check if it's L1 or L2 (Ln)
                if 'compL1' in filename or 'prodL1' in filename:
                    l1_files.append(filepath)
                elif 'compLn' in filename or 'prodLn' in filename:
                    l2_files.append(filepath)
    
    print(f"Found {len(l1_files)} L1 files and {len(l2_files)} L2 files")
    return l1_files, l2_files

# Step 3: Process each file individually and extract summary statistics
def extract_features_safely(l1_files, l2_files, mask_file):
    """Extract features from fMRI files, processing each file individually"""
    # Create a masker object with the language mask
    masker = NiftiMasker(
        mask_img=mask_file,
        standardize=True,         # Z-score normalize the data
        smoothing_fwhm=6,         # 6mm smoothing to improve SNR
        t_r=2.0,                  # Repetition time (adjust if needed)
        memory='nilearn_cache',   # Cache computations
        memory_level=1,
        verbose=1
    )
    
    # Fit the masker on the mask image
    masker.fit()
    
    # Initialize lists to store features and labels
    all_features = []
    all_labels = []
    
    # Process L1 files
    print("Processing L1 files...")
    for file_path in l1_files:
        try:
            # Load the 4D image
            img = image.load_img(file_path)
            
            # Check if it's a 4D image
            if len(img.shape) == 4:
                # Calculate mean across time
                mean_img = image.mean_img(img)
                
                # Extract features from the mean image
                features = masker.transform(mean_img)
                
                # Add to our list
                all_features.append(features[0])  # Extract the single row
                all_labels.append(0)  # 0 for L1
                
                print(f"Processed {file_path}")
            else:
                # Already a 3D image, just extract features
                features = masker.transform(img)
                all_features.append(features[0])
                all_labels.append(0)
                
                print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process L2 files
    print("Processing L2 files...")
    for file_path in l2_files:
        try:
            # Load the 4D image
            img = image.load_img(file_path)
            
            # Check if it's a 4D image
            if len(img.shape) == 4:
                # Calculate mean across time
                mean_img = image.mean_img(img)
                
                # Extract features from the mean image
                features = masker.transform(mean_img)
                
                # Add to our list
                all_features.append(features[0])  # Extract the single row
                all_labels.append(1)  # 1 for L2
                
                print(f"Processed {file_path}")
            else:
                # Already a 3D image, just extract features
                features = masker.transform(img)
                all_features.append(features[0])
                all_labels.append(1)
                
                print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"Successfully processed {len(all_features)} files out of {len(l1_files) + len(l2_files)} total files")
    print(f"Features shape: {features_array.shape}")
    
    return features_array, labels_array, masker

# Step 4: Train a simple classifier
def train_classifier(features, labels):
    """Train a simple SVM classifier with train-test split"""
    # Split data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    
    # Train a linear SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    
    # Test the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Classifier accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['L1', 'L2']))
    
    return clf, X_test, y_test, y_pred

# Step 5: Visualize the most discriminative brain regions
def visualize_weights(clf, masker):
    """Visualize the classifier weights to see which brain regions are important"""
    # Get the classifier weights
    weights = clf.coef_[0]
    
    # Transform weights back to brain space
    weight_img = masker.inverse_transform(weights)
    
    # Save the weight map
    weight_img.to_filename('l1_vs_l2_weights.nii.gz')
    
    # Plot the weight map
    print("Plotting the classifier weights - positive values (red) are more active for L2, "
          "negative values (blue) are more active for L1")
    
    plotting.plot_stat_map(
        weight_img,
        display_mode='ortho',
        title='Brain regions distinguishing L1 from L2'
    )
    plotting.show()

# Main function to run everything
def main():
    # Collect the functional files
    l1_files, l2_files = collect_functional_files(data_root)
    
    # Extract features safely, handling different FOVs
    features, labels, masker = extract_features_safely(l1_files, l2_files, mask_file)
    
    # Check if we have enough data to proceed
    if len(features) < 10:
        print("Not enough data successfully processed. Exiting.")
        return
    
    # Train classifier
    clf, X_test, y_test, y_pred = train_classifier(features, labels)
    
    # Visualize the classifier weights
    visualize_weights(clf, masker)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()