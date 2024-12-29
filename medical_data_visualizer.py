import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import the data from medical_examination.csv and assign it to the df variable. (ok)
df = pd.read_csv('medical_examination.csv')

# 2 Add an overweight column to the data. To determine if a person is overweight, first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3 Normalize os dados, tornando 0 sempre bom e 1 sempre ruim. Se o valor de cholesterol ou de gluc for 1, defina o valor como 0. Se o valor for maior que 1, defina o valor como 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4 Draw the Categorical Plot in the draw_cat_plot function.
def draw_cat_plot():
    # 5 Crie um DataFrame para o gráfico de categorias usando pd.melt com valores de cholesterol, gluc, smoke, alco, active e overweight na variável df_cat.
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6 Agrupe e reformate os dados em df_cat para dividi-los por cardio. Mostre as contagens de cada recurso. Você terá que renomear uma das colunas para que o catplot funcione corretamente.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # 7 Convert the data into long format and create a chart that shows the value counts of the categorical features using the following method provided by the seaborn library import: sns.catplot().
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=5, aspect=1
    ).fig

    # 8 Get the figure for the output and store it in the fig variable.
    fig.savefig('catplot.png')
    return fig

# 10 Draw the Heat Map in the draw_heat_map function.
def draw_heat_map():
    # 11 Limpe os dados na variável df_heat filtrando os seguintes segmentos de pacientes que representam dados incorretos:
    # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    # height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    # height is more than the 97.5th percentile
    # weight is less than the 2.5th percentile
    # weight is more than the 97.5th percentile
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 Calculate the correlation matrix and store it in the corr variable.
    corr = df_heat.corr()

    # 13 Generate a mask for the upper triangle and store it in the mask variable.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap().
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", square=True, cmap='coolwarm', cbar_kws={'shrink': 0.5}, ax=ax
    )

    # 16 Do not modify the next two lines.
    fig.savefig('heatmap.png')
    return fig
