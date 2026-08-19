import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import argparse

def plot_scores_34(input_csv: str, output_img: str):
    """Generates an ordered bar chart designed to display 34 peptides."""
    
    # 1. Load data
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {input_csv}")
        return

    # 2. Cleaning and Sorting
    if 'pep_sequence' not in df.columns or 'prediction_score' not in df.columns:
        print("ERROR: The CSV must contain the columns 'pep_sequence' and 'prediction_score'.")
        return

    # Sort from highest to lowest score to prioritize candidates
    df_sorted = df.sort_values(by='prediction_score', ascending=False).reset_index(drop=True)
    
    # Use the full peptide sequences as X-axis labels (if they are short)
    df_sorted['Peptide_Label'] = df_sorted['pep_sequence']

    # 3. Plot Generation
    # Adjust figsize to fit 34 bars comfortably
    plt.figure(figsize=(18, 7)) 
    
    # Use a colormap that emphasizes high values (best candidates)
    palette = sns.color_palette("viridis", n_colors=len(df_sorted))
    
    sns.barplot(
        x='Peptide_Label', 
        y='prediction_score', 
        data=df_sorted, 
        palette=palette,
        hue='Peptide_Label',  # Added to prevent Seaborn deprecation warnings
        legend=False          # Added to prevent Seaborn deprecation warnings
    )

    # Add classification threshold line (0.5)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Strong/Weak Threshold (0.5)')

    # 4. Visual Adjustments
    plt.title('Interaction Prediction Scores (34 ESM Candidate Peptides)', fontsize=16)
    plt.xlabel('Candidate Peptide (Sequence)', fontsize=12)
    plt.ylabel('Strong Interaction Probability (Score)', fontsize=12)
    
    # Rotate labels 60 degrees to avoid overlap among the 34 sequences
    plt.xticks(rotation=60, ha='right', fontsize=9) 
    plt.ylim(0, 1) 
    
    # We create a custom legend for the threshold line
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, loc='upper right')
        
    plt.tight_layout()
    
    # 5. Save and Download
    plt.savefig(output_img, dpi=300)
    print(f"\nPlot saved at: {output_img}")
    
    try:
        files.download(output_img)
    except Exception as e:
        print(f"Could not initiate download. Please download the file {output_img} manually.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generates prediction scores plot.")
    ap.add_argument("--input-csv", required=True, help="Path to the CSV file with the final predictions.")
    ap.add_argument("--output-img", default="pdi_final_scores_34.png", help="Output image file name.")
    
    args = ap.parse_args()
    plot_scores_34(args.input_csv, args.output_img)
