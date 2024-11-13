import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def winsorize(df, column, upper, lower):
    col_df = df[column]
    
    perc_upper = np.percentile(df[column],upper)
    perc_lower = np.percentile(df[column],lower)
    
    df[column] = np.where(df[column] >= perc_upper, 
                          perc_upper, 
                          df[column])
    
    df[column] = np.where(df[column] <= perc_lower, 
                          perc_lower, 
                          df[column])
    
    return df

def heatmap_corr(df, title):    
    plt.figure(figsize=(20, 10))
    plt.title(title, fontsize=20, fontweight='bold')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"shrink": .8})
    plt.xticks(ticks=range(len(df.columns)), labels=df.columns, rotation=45, ha='right')
    plt.yticks(ticks=range(len(df.columns)), labels=df.columns, rotation=0)
    plt.tight_layout()
    plt.show()