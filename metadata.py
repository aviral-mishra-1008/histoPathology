from configs import *

def get_metadata():
    # --- Download and Filter Metadata ---
    try:
        metadata_file_path = hf_hub_download(
            repo_id=metadata_repo_id,
            filename="metadata.json",
            repo_type="dataset",
            cache_dir="./metadata_cache"
        )

        with open(metadata_file_path, 'r') as f:
            master_metadata = json.load(f)

        target_dataset_name = "HISTAI-colorectal-b2"

        # Filter metadata based on 'case_mapping' field and use case identifier from case_mapping as the key
        filtered_metadata = {}
        for entry in master_metadata:
            case_mapping = entry.get('case_mapping')
            if case_mapping and target_dataset_name in case_mapping:
                # Extract case identifier from case_mapping (e.g., 'case_58')
                check = entry.get('icd10')
                if 'NONE' in check or ',' in check:
                    continue
                case_identifier = case_mapping.split('/')[-1]
                filtered_metadata[case_identifier] = entry

        print(f)
        print(f"Successfully loaded metadata and filtered for '{target_dataset_name}'.")
        print(f"Found {len(filtered_metadata)} cases with metadata.")

        if filtered_metadata:
            sample_case_id = list(filtered_metadata.keys())[0]
            sample_entry = filtered_metadata[sample_case_id]
            print("\nSample metadata entry:")
            for key, value in sample_entry.items():
                print(f"  - {key}: {value}")

    except Exception as e:
        print(f"Error downloading or loading metadata: {e}")
        filtered_metadata = {}


    # Get the list of WSI files from the colorectal repository
    api = HfApi()
    file_list = api.list_repo_files(repo_id=wsi_repo_id, repo_type="dataset")
    wsi_files = [f for f in file_list if f.endswith('.tiff')]

    # Filter WSI files to include only those with corresponding metadata
    wsi_files_with_metadata = []
    for f in wsi_files:
        if f.split('/')[0] in filtered_metadata:
            wsi_files_with_metadata.append(f)

    print(f"\nFound {len(wsi_files)} WSI files in the {wsi_repo_id} repository.")
    print(f"Using {len(wsi_files_with_metadata)} WSI files that have corresponding metadata.")

    return wsi_files_with_metadata,filtered_metadata