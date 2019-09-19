import seaborn as sns
sns.set(style="whitegrid")


def barplot(df):
    tips = sns.load_dataset("tips")
    ax = sns.barplot(x="day", y="total_bill", data=tips)
