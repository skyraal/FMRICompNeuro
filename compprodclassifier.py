import os
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
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

# Step 2: Collect all relevant functional files, organized by subject ID
def collect_functional_files(data_root):
    """Collect all functional files and organize by subject ID, L1/L2 and task type"""
    # Dictionary to store files by subject ID
    files_by_subject = {}
    
    # Loop through all subject folders
    for subj_id in range(1, 35):  # Subjects 01-34
        subj_folder = os.path.join(data_root, f'sub-{subj_id:02d}')
        func_folder = os.path.join(subj_folder, 'func')
        
        if not os.path.exists(func_folder):
            print(f"Warning: No func folder for subject {subj_id}")
            continue
        
        # Initialize subject entry with empty lists for each task/language combination
        files_by_subject[subj_id] = {
            'comprehension': {'L1': [], 'L2': []},
            'production': {'L1': [], 'L2': []}
        }
        
        # Look for all functional files
        for filename in os.listdir(func_folder):
            if filename.endswith('_bold.nii.gz'):  # Only get the BOLD files
                filepath = os.path.join(func_folder, filename)
                
                # Categorize by language and task
                if 'compL1' in filename:
                    files_by_subject[subj_id]['comprehension']['L1'].append(filepath)
                elif 'compLn' in filename:
                    files_by_subject[subj_id]['comprehension']['L2'].append(filepath)
                elif 'prodL1' in filename:
                    files_by_subject[subj_id]['production']['L1'].append(filepath)
                elif 'prodLn' in filename:
                    files_by_subject[subj_id]['production']['L2'].append(filepath)
    
    # Count total files by category
    comp_l1_count = sum(len(s['comprehension']['L1']) for s in files_by_subject.values())
    comp_l2_count = sum(len(s['comprehension']['L2']) for s in files_by_subject.values())
    prod_l1_count = sum(len(s['production']['L1']) for s in files_by_subject.values())
    prod_l2_count = sum(len(s['production']['L2']) for s in files_by_subject.values())
    
    print(f"Found {comp_l1_count} L1 comprehension files and {comp_l2_count} L2 comprehension files")
    print(f"Found {prod_l1_count} L1 production files and {prod_l2_count} L2 production files")
    print(f"Found data for {len(files_by_subject)} subjects")
    
    return files_by_subject

# Step 3: Extract features while tracking subject IDs
def extract_features(files_by_subject, task_type, mask_file):
    """Extract features while preserving subject identities"""
    masker = NiftiMasker(
        mask_img=mask_file,
        standardize=True,
        smoothing_fwhm=6,
        t_r=2.0,
        memory='nilearn_cache',
        memory_level=1,
        verbose=1
    )
    
    # Fit the masker
    masker.fit()
    
    # Initialize arrays to store features, labels, and subject IDs
    all_features = []
    all_labels = []    # 0 for L1, 1 for L2
    all_subjects = []  # Store subject IDs for each sample
    
    # Process all subjects
    for subject_id, subject_data in files_by_subject.items():
        # Process L1 files
        for file_path in subject_data[task_type]['L1']:
            try:
                img = image.load_img(file_path)
                # Calculate mean across time if 4D
                if len(img.shape) == 4:
                    mean_img = image.mean_img(img)
                    features = masker.transform(mean_img)
                else:
                    features = masker.transform(img)
                
                all_features.append(features[0])
                all_labels.append(0)  # 0 for L1
                all_subjects.append(subject_id)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Process L2 files
        for file_path in subject_data[task_type]['L2']:
            try:
                img = image.load_img(file_path)
                # Calculate mean across time if 4D
                if len(img.shape) == 4:
                    mean_img = image.mean_img(img)
                    features = masker.transform(mean_img)
                else:
                    features = masker.transform(img)
                
                all_features.append(features[0])
                all_labels.append(1)  # 1 for L2
                all_subjects.append(subject_id)
                #print(f"Processed L2 {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    subjects_array = np.array(all_subjects)
    
    print(f"Features shape: {features_array.shape}")
    
    return features_array, labels_array, subjects_array, masker

# Step 4: Train a classifier with subject-level cross-validation
def train_classifier(features, labels, subjects, task_name):
    """Train a classifier ensuring no subject appears in both training and test sets"""
    # Get unique subject IDs
    unique_subjects = np.unique(subjects)
    
    # Split SUBJECTS (not samples) into training and testing sets
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=42
    )
    
    # Create masks for training and test samples based on subject assignment
    train_mask = np.isin(subjects, train_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    # Apply masks to get training and test sets
    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]
    
    print(f"Training {task_name} classifier on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    print(f"Training subjects: {sorted(train_subjects)}")
    print(f"Testing subjects: {sorted(test_subjects)}")
    
    # Train a linear SVM
    #clf = SVC(kernel='linear', C=1.0)
    #clf = LogisticRegression(C=1.0, max_iter = 1000)
    clf = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, random_state=42)
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
    """Visualize the classifier weights to show important brain regions"""
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
def run_task_analysis(task_name, files_by_subject, mask_file):
    print(f"\n\n=== Running analysis for {task_name} tasks ===\n")
    
    # Extract features while preserving subject information
    features, labels, subjects, masker = extract_features(files_by_subject, task_name, mask_file)
    
    # Check if we have enough data to proceed
    if len(features) < 10:
        print(f"Not enough data successfully processed for {task_name} tasks. Skipping.")
        return
    
    # Train classifier with subject-level split
    clf, X_test, y_test, y_pred = train_classifier(features, labels, subjects, task_name)
    
    # Visualize the classifier weights
    visualize_weights(clf, masker, task_name)
    
    print(f"{task_name.capitalize()} analysis complete!")
    
    return clf

# Main function to run everything
def main():
    # Collect the functional files organized by subject
    files_by_subject = collect_functional_files(data_root)
    
    # Run separate analyses for comprehension and production tasks
    comp_clf = run_task_analysis('comprehension', files_by_subject, mask_file)
    prod_clf = run_task_analysis('production', files_by_subject, mask_file)
    
    print("\nBoth analyses complete!")
    
    # Compare the two classifiers
    if comp_clf and prod_clf:
        print("\nAnalysis complete! You now have separate L1 vs L2 classifiers for:")
        print("1. Language comprehension tasks")
        print("2. Language production tasks")
        print("\nThe weight maps have been saved as:")
        print("- l1_vs_l2_comprehension_weights.nii.gz")
        print("- l1_vs_l2_production_weights.nii.gz")

if __name__ == "__main__":
    main()