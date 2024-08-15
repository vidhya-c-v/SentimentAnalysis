mean_sentiments = user_data.groupby('ticker')['compound'].mean()

    # Plot bar chart of mean sentiment scores
    plt.figure(figsize=(10, 6))
    colors = ['skyblue' if x >= 0 else 'lightcoral' for x in mean_sentiments]
    plt.bar(mean_sentiments.index, mean_sentiments, color=colors)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Neutral Line')