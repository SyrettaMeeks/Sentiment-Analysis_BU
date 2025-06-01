# Sentiment-Analysis_BU
Innovation Product Survey Results
# Complete Sentiment Analysis Tutorial in Python
# From beginner to analyzing your survey data

# ============================================================================
# STEP 1: Install Required Libraries
# ============================================================================
# Run these commands in your terminal/command prompt:
# pip install pandas textblob vaderSentiment matplotlib seaborn

# ============================================================================
# STEP 2: Import Libraries
# ============================================================================
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 3: Create Your Survey Data
# ============================================================================
# Let's recreate your survey data
survey_data = {
    'timestamp': [
        '5/24/2025 15:08:06', '5/28/2025 19:08:50', '5/28/2025 19:09:00',
        '5/28/2025 19:12:34', '5/30/2025 21:09:16', '5/30/2025 21:33:53',
        '5/30/2025 21:38:55', '5/30/2025 22:05:00', '5/31/2025 12:22:42',
        '5/31/2025 12:25:09', '5/31/2025 13:49:52', '5/31/2025 15:50:17',
        '5/31/2025 17:17:39', '5/31/2025 17:45:56', '5/31/2025 18:32:23',
        '5/31/2025 21:44:58'
    ],
    'rating': [10, 9, 9, 9, 9, 10, 10, 10, 10, 8, 9, 9, 10, 10, 10, 8],
    'description': [
        'Utility', 'Helpful', 'useful and innovative', 'useful', 'shocked',
        'Convenient', 'Ingenious', 'Keep this up I love this', 'Convenient',
        'Multifaceted', 'Prompt', 'Excited', 'Relevant', 'Enthusiastic',
        'Innovative', 'Like the generators'
    ],
    'willingness_to_pay': [
        'Yes, $39/Month', 'Yes, 50 monthly', 'definitely, under 100 a month',
        '100$', 'The basic or business', 'Yes, $25/month',
        'Yes. $450.00 annually: business pro option', 'No',
        'The asking price is reasonable.', 'business tier', '35.00 a month',
        'I would. Depending on the sophistication and exec', '$100',
        'Yes, any price.', 'Yes, I would be willing to pay for this!',
        'Yes but only for the basic until I can see savings a'
    ]
}

# Create DataFrame
df = pd.DataFrame(survey_data)
print("Your Survey Data:")
print(df.head())
print(f"\nTotal responses: {len(df)}")

# ============================================================================
# STEP 4: Basic Sentiment Analysis with TextBlob
# ============================================================================
print("\n" + "="*60)
print("STEP 4: TEXTBLOB SENTIMENT ANALYSIS")
print("="*60)

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob
    Returns polarity (-1 to 1) and subjectivity (0 to 1)
    """
    blob = TextBlob(str(text))
    return {
        'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
        'subjectivity': blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
    }

# Apply TextBlob analysis
textblob_results = df['description'].apply(analyze_sentiment_textblob)
df['textblob_polarity'] = [result['polarity'] for result in textblob_results]
df['textblob_subjectivity'] = [result['subjectivity'] for result in textblob_results]

# Add sentiment labels
def get_sentiment_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['textblob_sentiment'] = df['textblob_polarity'].apply(get_sentiment_label)

print("TextBlob Results:")
print(df[['description', 'textblob_polarity', 'textblob_sentiment']].head(10))

# ============================================================================
# STEP 5: VADER Sentiment Analysis
# ============================================================================
print("\n" + "="*60)
print("STEP 5: VADER SENTIMENT ANALYSIS")
print("="*60)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER
    Returns compound score (-1 to 1) and individual scores
    """
    scores = analyzer.polarity_scores(str(text))
    return scores

# Apply VADER analysis
vader_results = df['description'].apply(analyze_sentiment_vader)
df['vader_compound'] = [result['compound'] for result in vader_results]
df['vader_positive'] = [result['pos'] for result in vader_results]
df['vader_negative'] = [result['neg'] for result in vader_results]
df['vader_neutral'] = [result['neu'] for result in vader_results]

# Add VADER sentiment labels
def get_vader_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['vader_sentiment'] = df['vader_compound'].apply(get_vader_sentiment)

print("VADER Results:")
print(df[['description', 'vader_compound', 'vader_sentiment']].head(10))

# ============================================================================
# STEP 6: Compare Methods
# ============================================================================
print("\n" + "="*60)
print("STEP 6: COMPARISON OF METHODS")
print("="*60)

comparison_df = df[['description', 'rating', 'textblob_sentiment', 'vader_sentiment']].copy()
print("Side-by-Side Comparison:")
print(comparison_df)

# Check agreement between methods
agreement = (df['textblob_sentiment'] == df['vader_sentiment']).sum()
total = len(df)
print(f"\nAgreement between TextBlob and VADER: {agreement}/{total} ({agreement/total*100:.1f}%)")

# ============================================================================
# STEP 7: Advanced Analysis - Correlation with Ratings
# ============================================================================
print("\n" + "="*60)
print("STEP 7: CORRELATION WITH SURVEY RATINGS")
print("="*60)

# Calculate correlations
textblob_corr = df['textblob_polarity'].corr(df['rating'])
vader_corr = df['vader_compound'].corr(df['rating'])

print(f"Correlation between TextBlob polarity and rating: {textblob_corr:.3f}")
print(f"Correlation between VADER compound and rating: {vader_corr:.3f}")

# Group by sentiment and see average ratings
print("\nAverage ratings by sentiment (TextBlob):")
sentiment_ratings = df.groupby('textblob_sentiment')['rating'].agg(['mean', 'count'])
print(sentiment_ratings)

# ============================================================================
# STEP 8: Extract Pricing Willingness
# ============================================================================
print("\n" + "="*60)
print("STEP 8: ANALYZE PRICING SENTIMENT")
print("="*60)

# Analyze sentiment of pricing responses
pricing_sentiment = df['willingness_to_pay'].apply(analyze_sentiment_textblob)
df['pricing_polarity'] = [result['polarity'] for result in pricing_sentiment]
df['pricing_sentiment'] = df['pricing_polarity'].apply(get_sentiment_label)

print("Pricing Sentiment Analysis:")
pricing_analysis = df[['willingness_to_pay', 'pricing_polarity', 'pricing_sentiment']]
print(pricing_analysis)

# ============================================================================
# STEP 9: Create Visualizations
# ============================================================================
print("\n" + "="*60)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*60)

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Sentiment Distribution
sentiment_counts = df['textblob_sentiment'].value_counts()
axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
axes[0, 0].set_title('Sentiment Distribution (TextBlob)')

# Plot 2: Sentiment vs Rating
sns.boxplot(data=df, x='textblob_sentiment', y='rating', ax=axes[0, 1])
axes[0, 1].set_title('Rating by Sentiment')

# Plot 3: Sentiment Scores Over Time
df['timestamp'] = pd.to_datetime(df['timestamp'])
axes[1, 0].plot(df['timestamp'], df['textblob_polarity'], marker='o')
axes[1, 0].set_title('Sentiment Over Time')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: TextBlob vs VADER comparison
axes[1, 1].scatter(df['textblob_polarity'], df['vader_compound'])
axes[1, 1].set_xlabel('TextBlob Polarity')
axes[1, 1].set_ylabel('VADER Compound')
axes[1, 1].set_title('TextBlob vs VADER Comparison')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 10: Summary Report
# ============================================================================
print("\n" + "="*60)
print("STEP 10: SUMMARY REPORT")
print("="*60)

print("SENTIMENT ANALYSIS SUMMARY:")
print("-" * 40)
print(f"Total Responses: {len(df)}")
print(f"Positive Sentiment: {(df['textblob_sentiment'] == 'Positive').sum()} ({(df['textblob_sentiment'] == 'Positive').mean()*100:.1f}%)")
print(f"Neutral Sentiment: {(df['textblob_sentiment'] == 'Neutral').sum()} ({(df['textblob_sentiment'] == 'Neutral').mean()*100:.1f}%)")
print(f"Negative Sentiment: {(df['textblob_sentiment'] == 'Negative').sum()} ({(df['textblob_sentiment'] == 'Negative').mean()*100:.1f}%)")

print(f"\nAverage Rating: {df['rating'].mean():.2f}")
print(f"Average Sentiment Score: {df['textblob_polarity'].mean():.3f}")

print("\nMOST POSITIVE RESPONSES:")
most_positive = df.nlargest(3, 'textblob_polarity')[['description', 'textblob_polarity', 'rating']]
print(most_positive.to_string(index=False))

print("\nMOST NEGATIVE RESPONSES:")
most_negative = df.nsmallest(3, 'textblob_polarity')[['description', 'textblob_polarity', 'rating']]
print(most_negative.to_string(index=False))

# ============================================================================
# STEP 11: Export Results
# ============================================================================
print("\n" + "="*60)
print("STEP 11: EXPORT RESULTS")
print("="*60)

# Save results to CSV
output_file = 'sentiment_analysis_results.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Create a summary DataFrame
summary_stats = {
    'Metric': ['Total Responses', 'Positive %', 'Neutral %', 'Negative %', 
               'Avg Rating', 'Avg Sentiment', 'TextBlob-VADER Agreement %'],
    'Value': [
        len(df),
        f"{(df['textblob_sentiment'] == 'Positive').mean()*100:.1f}%",
        f"{(df['textblob_sentiment'] == 'Neutral').mean()*100:.1f}%",
        f"{(df['textblob_sentiment'] == 'Negative').mean()*100:.1f}%",
        f"{df['rating'].mean():.2f}",
        f"{df['textblob_polarity'].mean():.3f}",
        f"{agreement/total*100:.1f}%"
    ]
}

summary_df = pd.DataFrame(summary_stats)
print("\nSUMMARY STATISTICS:")
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("TUTORIAL COMPLETE!")
print("="*60)
print("You now know how to:")
print("1. Set up sentiment analysis libraries")
print("2. Analyze text with TextBlob and VADER")
print("3. Compare different sentiment methods")
print("4. Correlate sentiment with other metrics")
print("5. Visualize sentiment data")
print("6. Export and summarize results")
