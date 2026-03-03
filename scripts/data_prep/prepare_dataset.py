import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_kfold_splits(parquet_path, n_splits=5, output_dir='kfold_splits'):
    # Load the dataset
    df = pd.read_parquet(parquet_path)

    # Group by book_id to ensure book-level separation
    book_df = df.groupby('book_id').agg({
        'printshop': 'first',  # Get the printshop for each book
        'image_path': 'count'  # Count pages per book
    }).reset_index()

    # Get all unique printshops
    all_printshops = df['printshop'].unique().tolist()
    print(f"Found {len(all_printshops)} unique printshops: {all_printshops}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate folds ensuring all printshops are represented in each fold
    fold_info = {}

    # Use stratified k-fold to get initial split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(book_df, book_df['printshop'])):
        # Get book_ids for train and test
        train_book_ids = book_df.iloc[train_idx]['book_id'].tolist()
        test_book_ids = book_df.iloc[test_idx]['book_id'].tolist()

        # Check if test set has all printshops
        test_printshops = set(book_df[book_df['book_id'].isin(test_book_ids)]['printshop'])
        missing_printshops = set(all_printshops) - test_printshops

        # If test set is missing printshops, move some books from train to test
        if missing_printshops:
            print(f"Fold {fold}: Test set missing printshops: {missing_printshops}")
            for printshop in missing_printshops:
                # Find books from this printshop in train set
                printshop_books = book_df[(book_df['book_id'].isin(train_book_ids)) &
                                         (book_df['printshop'] == printshop)]['book_id'].tolist()

                if printshop_books:
                    # Move one book to test set
                    book_to_move = printshop_books[0]
                    test_book_ids.append(book_to_move)
                    train_book_ids.remove(book_to_move)
                    print(f"  Moved book {book_to_move} from train to test set")

        # Further split train into train and validation
        train_books_df = book_df[book_df['book_id'].isin(train_book_ids)]

        # Check if we have enough books for each printshop in train set
        train_printshops_count = train_books_df['printshop'].value_counts().to_dict()

        # Ensure we have at least 2 books per printshop in train set if possible
        for printshop in all_printshops:
            if printshop not in train_printshops_count or train_printshops_count[printshop] < 2:
                print(f"Fold {fold}: Not enough books for {printshop} in train set")

        # Use stratified split for validation
        val_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx_inner, val_idx_inner = next(val_skf.split(train_books_df, train_books_df['printshop']))

        val_book_ids = train_books_df.iloc[val_idx_inner]['book_id'].tolist()
        train_book_ids = [bid for bid in train_book_ids if bid not in val_book_ids]

        # Check if validation set has all printshops
        val_printshops = set(book_df[book_df['book_id'].isin(val_book_ids)]['printshop'])
        missing_val_printshops = set(all_printshops) - val_printshops

        # If validation set is missing printshops, move some books from train to validation
        if missing_val_printshops:
            print(f"Fold {fold}: Validation set missing printshops: {missing_val_printshops}")
            for printshop in missing_val_printshops:
                # Find books from this printshop in train set
                printshop_books = book_df[(book_df['book_id'].isin(train_book_ids)) &
                                         (book_df['printshop'] == printshop)]['book_id'].tolist()

                if printshop_books:
                    # Move one book to validation set
                    book_to_move = printshop_books[0]
                    val_book_ids.append(book_to_move)
                    train_book_ids.remove(book_to_move)
                    print(f"  Moved book {book_to_move} from train to validation set")

        # Verify printshop distribution in each split
        train_printshops = set(book_df[book_df['book_id'].isin(train_book_ids)]['printshop'])
        val_printshops = set(book_df[book_df['book_id'].isin(val_book_ids)]['printshop'])
        test_printshops = set(book_df[book_df['book_id'].isin(test_book_ids)]['printshop'])

        print(f"Fold {fold} distribution:")
        print(f"  Train: {len(train_book_ids)} books, printshops: {train_printshops}")
        print(f"  Validation: {len(val_book_ids)} books, printshops: {val_printshops}")
        print(f"  Test: {len(test_book_ids)} books, printshops: {test_printshops}")

        # Store fold information
        fold_info[f'fold_{fold}'] = {
            'train': train_book_ids,
            'val': val_book_ids,
            'test': test_book_ids
        }

        # Save fold information
        fold_file = os.path.join(output_dir, f'fold_{fold}.json')
        print(f"Saving fold information to {fold_file}")
        with open(fold_file, 'w') as f:
            json.dump({
                'train': train_book_ids,
                'val': val_book_ids,
                'test': test_book_ids
            }, f, indent=2)

        # Verify the file was created
        if os.path.exists(fold_file):
            print(f"Successfully created {fold_file}")
        else:
            print(f"WARNING: Failed to create {fold_file}")

        # Also save printshop distribution information
        with open(os.path.join(output_dir, f'fold_{fold}_stats.json'), 'w') as f:
            train_counts = book_df[book_df['book_id'].isin(train_book_ids)]['printshop'].value_counts().to_dict()
            val_counts = book_df[book_df['book_id'].isin(val_book_ids)]['printshop'].value_counts().to_dict()
            test_counts = book_df[book_df['book_id'].isin(test_book_ids)]['printshop'].value_counts().to_dict()

            json.dump({
                'train_counts': train_counts,
                'val_counts': val_counts,
                'test_counts': test_counts,
                'train_printshops': list(train_printshops),
                'val_printshops': list(val_printshops),
                'test_printshops': list(test_printshops)
            }, f, indent=2)

    # Save all folds information
    with open(os.path.join(output_dir, 'all_folds.json'), 'w') as f:
        json.dump(fold_info, f, indent=2)

    # Create a visualization of the fold distribution
    try:
        visualize_fold_distribution(book_df, fold_info, output_dir)
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")

    return fold_info

def visualize_fold_distribution(book_df, fold_info, output_dir):
    """Create visualizations of the printshop distribution across folds"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up the figure
    plt.figure(figsize=(15, 10))

    # Create a DataFrame to hold the distribution data
    distribution_data = []

    for fold_name, fold_data in fold_info.items():
        for split_name, book_ids in fold_data.items():
            # Get printshop counts for this split
            printshop_counts = book_df[book_df['book_id'].isin(book_ids)]['printshop'].value_counts().to_dict()

            # Add to our data
            for printshop, count in printshop_counts.items():
                distribution_data.append({
                    'Fold': fold_name,
                    'Split': split_name,
                    'Printshop': printshop,
                    'Count': count
                })

    # Convert to DataFrame
    dist_df = pd.DataFrame(distribution_data)

    # Create the plot
    sns.barplot(x='Fold', y='Count', hue='Printshop', data=dist_df)
    plt.title('Distribution of Printshops Across Folds')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'fold_distribution.png'))

    # Create a second plot showing distribution by split type
    plt.figure(figsize=(15, 10))
    g = sns.catplot(
        data=dist_df, kind="bar",
        x="Printshop", y="Count", hue="Split", col="Fold",
        height=4, aspect=.7, palette="muted",
        legend_out=False
    )
    g.set_xticklabels(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_split_distribution.png'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare k-fold splits for document classification dataset')
    parser.add_argument('--parquet_path', type=str, required=True, help='Path to the parquet file')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--output_dir', type=str, default='kfold_splits', help='Output directory for fold information')

    args = parser.parse_args()
    prepare_kfold_splits(args.parquet_path, args.n_splits, args.output_dir)
