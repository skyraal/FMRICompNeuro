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

# Set up paths and parameters
data_root = './adultlanglearn_dataset/'
derivatives_dir = os.path.join(data_root, 'derivatives')
mask_file = os.path.join(derivatives_dir, 'mask', 'language.img')

# Step 2: Collect all relevant functional files, separated by task type
def collect_functional_files(data_root):
    """Collect all functional files and organize by L1/L2 and task type (comprehension/production)"""
    l1_comp_files = []  # L1 comprehension files
    l2_comp_files = []  # L2 comprehension files
    l1_prod_files = []  # L1 production files
    l2_prod_files = []  # L2 production files
    
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
                
                # Categorize by language and task
                if 'compL1' in filename:
                    l1_comp_files.append(filepath)
                elif 'compLn' in filename:
                    l2_comp_files.append(filepath)
                elif 'prodL1' in filename:
                    l1_prod_files.append(filepath)
                elif 'prodLn' in filename:
                    l2_prod_files.append(filepath)
    
    print(f"Found {len(l1_comp_files)} L1 comprehension files and {len(l2_comp_files)} L2 comprehension files")
    print(f"Found {len(l1_prod_files)} L1 production files and {len(l2_prod_files)} L2 production files")
    
    return {
        'comprehension': (l1_comp_files, l2_comp_files),
        'production': (l1_prod_files, l2_prod_files)
    }

# Step 3: Process each file individually and extract summary statistics
def extract_features_safely(l1_files, l2_files, mask_file, task_name):
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
    print(f"Processing L1 {task_name} files...")
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
    print(f"Processing L2 {task_name} files...")
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
def train_classifier(features, labels, task_name):
    """Train a simple SVM classifier with train-test split"""
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training {task_name} classifier on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    
    # Train a linear SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    
    # Test the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{task_name.capitalize()} Classifier accuracy: {accuracy:.2f}")
    print(f"\n{task_name.capitalize()} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['L1', 'L2']))
    
    return clf, X_test, y_test, y_pred

# Step 5: Visualize the most discriminative brain regions
def visualize_weights(clf, masker, task_name):
    """Visualize the classifier weights to see which brain regions are important"""
    # Get the classifier weights
    weights = clf.coef_[0]
    
    # Transform weights back to brain space
    weight_img = masker.inverse_transform(weights)
    
    # Save the weight map
    weight_img.to_filename(f'l1_vs_l2_{task_name}_weights.nii.gz')
    
    # Plot the weight map
    print(f"Plotting the {task_name} classifier weights - positive values (red) are more active for L2, "
          "negative values (blue) are more active for L1")
    
    plotting.plot_stat_map(
        weight_img,
        display_mode='ortho',
        title=f'Brain regions distinguishing L1 from L2 in {task_name} tasks'
    )
    plotting.show()

# Run analysis for a specific task type
def run_task_analysis(task_name, l1_files, l2_files, mask_file):
    print(f"\n\n=== Running analysis for {task_name} tasks ===\n")
    
    # Extract features safely, handling different FOVs
    features, labels, masker = extract_features_safely(l1_files, l2_files, mask_file, task_name)
    
    # Check if we have enough data to proceed
    if len(features) < 10:
        print(f"Not enough data successfully processed for {task_name} tasks. Skipping.")
        return
    
    # Train classifier
    clf, X_test, y_test, y_pred = train_classifier(features, labels, task_name)
    
    # Visualize the classifier weights
    visualize_weights(clf, masker, task_name)
    
    print(f"{task_name.capitalize()} analysis complete!")
    
    return clf

# Main function to run everything
def main():
    # Collect the functional files
    file_collections = collect_functional_files(data_root)
    
    # Run separate analyses for comprehension and production tasks
    # Comprehension analysis
    l1_comp_files, l2_comp_files = file_collections['comprehension']
    comp_clf = run_task_analysis('comprehension', l1_comp_files, l2_comp_files, mask_file)
    
    # Production analysis
    l1_prod_files, l2_prod_files = file_collections['production']
    prod_clf = run_task_analysis('production', l1_prod_files, l2_prod_files, mask_file)
    
    print("\nBoth analyses complete!")
    
    # Optional: Compare the two classifiers' weights to see differences
    if comp_clf and prod_clf:
        print("\nAnalysis complete! You now have separate L1 vs L2 classifiers for:")
        print("1. Language comprehension tasks")
        print("2. Language production tasks")
        print("\nThe weight maps have been saved as:")
        print("- l1_vs_l2_comprehension_weights.nii.gz")
        print("- l1_vs_l2_production_weights.nii.gz")

if __name__ == "__main__":
    main()
