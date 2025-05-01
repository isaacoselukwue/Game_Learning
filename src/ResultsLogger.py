import pandas as pd, os

#for my report logging
def append_rows(rows, csv_path="results_master.csv"):
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)

    # Pretty Markdown next to the CSV
    md_path = csv_path.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
