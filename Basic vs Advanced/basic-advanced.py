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
import csv
warnings.filterwarnings("ignore")

# Set up paths and parameters
data_root = '../adultlanglearn_dataset/'
derivatives_dir = os.path.join(data_root, 'derivatives')
mask_file = os.path.join(derivatives_dir, 'mask', 'language.img')

# Step 2: Load proficiency data from TSV
def load_proficiency_data(data_root):
    """Load proficiency level data for each subject from participants.tsv"""
    proficiency_levels = {}
    
    # Path to participants.tsv file
    participants_file = os.path.join(data_root, 'participants.tsv')
    
    # Load proficiency data from TSV file
    with open(participants_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Extract subject ID from format like "sub-01"
            subject_id = int(row['participant_id'].split('-')[1])
            proficiency = row['group']
            proficiency_levels[subject_id] = proficiency
    
    print(f"Loaded proficiency data for {len(proficiency_levels)} subjects")
    return proficiency_levels

# Step 3: Collect all relevant functional files, organized by subject ID
def collect_functional_files(data_root):
    """Collect all functional files and organize by subject ID, L1/L2 and proficiency level"""
    # Dictionary to store files by subject ID
    files_by_subject = {}
    
    # Load proficiency levels from TSV file
    proficiency_levels = load_proficiency_data(data_root)
    
    # Loop through all subject folders
    for subj_id in range(1, 35):  # Subjects 01-34
        subj_folder = os.path.join(data_root, f'sub-{subj_id:02d}')
        func_folder = os.path.join(subj_folder, 'func')
        
        if not os.path.exists(func_folder):
            print(f"Warning: No func folder for subject {subj_id}")
            continue
        
        # Get proficiency level for this subject
        proficiency = proficiency_levels.get(subj_id, 'basic')  # Default to basic if not found
        
        # Initialize subject entry with empty lists for each proficiency/language combination
        files_by_subject[subj_id] = {
            'proficiency': proficiency,
            'L1': [],
            'L2': []
        }
        
        # Look for all functional files
        for filename in os.listdir(func_folder):
            if filename.endswith('_bold.nii.gz'):  # Only get the BOLD files
                filepath = os.path.join(func_folder, filename)
                
                # Categorize by language (combining comprehension and production tasks)
                if 'compL1' in filename or 'prodL1' in filename:
                    files_by_subject[subj_id]['L1'].append(filepath)
                elif 'compLn' in filename or 'prodLn' in filename:
                    files_by_subject[subj_id]['L2'].append(filepath)
    
    # Count total files by category and proficiency
    advanced_l1_count = sum(len(s['L1']) for s in files_by_subject.values() if s['proficiency'] == 'advanced')
    advanced_l2_count = sum(len(s['L2']) for s in files_by_subject.values() if s['proficiency'] == 'advanced')
    basic_l1_count = sum(len(s['L1']) for s in files_by_subject.values() if s['proficiency'] == 'basic')
    basic_l2_count = sum(len(s['L2']) for s in files_by_subject.values() if s['proficiency'] == 'basic')
    
    print(f"Advanced group: Found {advanced_l1_count} L1 files and {advanced_l2_count} L2 files")
    print(f"Basic group: Found {basic_l1_count} L1 files and {basic_l2_count} L2 files")
    print(f"Found data for {len(files_by_subject)} subjects")
    
    return files_by_subject

# Step 4: Extract features while tracking subject IDs
def extract_features(files_by_subject, proficiency_level, mask_file):
    """Extract features for a specific proficiency level while preserving subject identities"""
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
    
    # Process subjects with the specified proficiency level
    for subject_id, subject_data in files_by_subject.items():
        # Skip subjects with different proficiency level
        if subject_data['proficiency'] != proficiency_level:
            continue
            
        # Process L1 files
        for file_path in subject_data['L1']:
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
        for file_path in subject_data['L2']:
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
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    subjects_array = np.array(all_subjects)
    
    print(f"Features shape for {proficiency_level} group: {features_array.shape}")
    
    return features_array, labels_array, subjects_array, masker

# Step 5: Train a classifier with subject-level cross-validation
def train_classifier(features, labels, subjects, proficiency_level):
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
    
    print(f"Training {proficiency_level} classifier on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    print(f"Training subjects: {sorted(train_subjects)}")
    print(f"Testing subjects: {sorted(test_subjects)}")
    

    #clf = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, random_state=42)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    
    # Test the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{proficiency_level.capitalize()} Classifier accuracy: {accuracy:.2f}")
    print(f"\n{proficiency_level.capitalize()} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['L1', 'L2']))
    
    return clf, X_test, y_test, y_pred

# Step 6: Visualize the most discriminative brain regions
def visualize_weights(clf, masker, proficiency_level):
    """Visualize the classifier weights to show important brain regions"""
    # Get the classifier weights
    weights = clf.coef_[0]
    
    # Transform weights back to brain space
    weight_img = masker.inverse_transform(weights)
    
    # Save the weight map
    weight_img.to_filename(f'l1_vs_l2_{proficiency_level}_weights.nii.gz')
    
    # Plot the weight map
    print(f"Plotting the {proficiency_level} classifier weights - positive values (red) are more active for L2, "
          "negative values (blue) are more active for L1")
    
    plotting.plot_stat_map(
        weight_img,
        display_mode='ortho',
        title=f'Brain regions distinguishing L1 from L2 in {proficiency_level} subjects'
    )
    plotting.show()

# Run analysis for a specific proficiency level
def run_proficiency_analysis(proficiency_level, files_by_subject, mask_file):
    print(f"\n\n=== Running analysis for {proficiency_level} proficiency level ===\n")
    
    # Extract features while preserving subject information
    features, labels, subjects, masker = extract_features(files_by_subject, proficiency_level, mask_file)
    
    # Check if we have enough data to proceed
    if len(features) < 10:
        print(f"Not enough data successfully processed for {proficiency_level} proficiency. Skipping.")
        return None
    
    # Train classifier with subject-level split
    clf, X_test, y_test, y_pred = train_classifier(features, labels, subjects, proficiency_level)
    
    # Visualize the classifier weights
    visualize_weights(clf, masker, proficiency_level)
    
    print(f"{proficiency_level.capitalize()} analysis complete!")
    
    return clf

# Main function to run everything
def main():
    # Collect the functional files organized by subject and proficiency
    files_by_subject = collect_functional_files(data_root)
    
    # Run separate analyses for advanced and basic proficiency levels
    advanced_clf = run_proficiency_analysis('advanced', files_by_subject, mask_file)
    basic_clf = run_proficiency_analysis('basic', files_by_subject, mask_file)
    
    print("\nBoth analyses complete!")
    
    # Compare the two classifiers
    if advanced_clf and basic_clf:
        print("\nAnalysis complete! You now have separate L1 vs L2 classifiers for:")
        print("1. Advanced L2 learners")
        print("2. Basic L2 learners")
        print("\nThe weight maps have been saved as:")
        print("- l1_vs_l2_advanced_weights.nii.gz")
        print("- l1_vs_l2_basic_weights.nii.gz")
        print("\nYou can compare these weight maps to see how L1/L2 processing differs in the brain")
        print("based on proficiency level.")

if __name__ == "__main__":
    main()