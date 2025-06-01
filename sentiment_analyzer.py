"""
Sentiment Analyzer for Business Understanding
A comprehensive class for analyzing sentiment in survey data and customer feedback.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis tool for business applications.
    
    This class provides methods to analyze sentiment using multiple approaches,
    correlate with business metrics, and generate insights for decision-making.
    """
    
    def __init__(self):
        """Initialize the SentimentAnalyzer with required analyzers."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.results = None
        
    def analyze_text_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with polarity and subjectivity scores
        """
        if pd.isna(text) or text == "":
            return {'polarity': 0, 'subjectivity': 0}
            
        blob = TextBlob(str(text))
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_text_vader(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary with compound, pos, neg, neu scores
        """
        if pd.isna(text) or text == "":
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
            
        return self.vader_analyzer.polarity_scores(str(text))
    
    def get_sentiment_label(self, polarity, method='textblob'):
        """
        Convert polarity score to sentiment label.
        
        Args:
            polarity (float): Polarity score
            method (str): Method used ('textblob' or 'vader')
            
        Returns:
            str: Sentiment label ('Positive', 'Negative', 'Neutral')
        """
        threshold = 0.05 if method == 'vader' else 0.1
        
        if polarity > threshold:
            return 'Positive'
        elif polarity < -threshold:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_dataframe(self, df, text_column, rating_column=None, 
                         timestamp_column=None, price_column=None):
        """
        Analyze sentiment for an entire DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Column name containing text to analyze
            rating_column (str, optional): Column name containing ratings
            timestamp_column (str, optional): Column name containing timestamps
            price_column (str, optional): Column name containing price feedback
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        results_df = df.copy()
        
        # TextBlob Analysis
        textblob_results = results_df[text_column].apply(self.analyze_text_textblob)
        results_df['textblob_polarity'] = [r['polarity'] for r in textblob_results]
        results_df['textblob_subjectivity'] = [r['subjectivity'] for r in textblob_results]
        results_df['textblob_sentiment'] = results_df['textblob_polarity'].apply(
            lambda x: self.get_sentiment_label(x, 'textblob')
        )
        
        # VADER Analysis
        vader_results = results_df[text_column].apply(self.analyze_text_vader)
        results_df['vader_compound'] = [r['compound'] for r in vader_results]
        results_df['vader_positive'] = [r['pos'] for r in vader_results]
        results_df['vader_negative'] = [r['neg'] for r in vader_results]
        results_df['vader_neutral'] = [r['neu'] for r in vader_results]
        results_df['vader_sentiment'] = results_df['vader_compound'].apply(
            lambda x: self.get_sentiment_label(x, 'vader')
        )
        
        # Analyze price column if provided
        if price_column and price_column in results_df.columns:
            price_sentiment = results_df[price_column].apply(self.analyze_text_textblob)
            results_df['price_sentiment_polarity'] = [r['polarity'] for r in price_sentiment]
            results_df['price_sentiment_label'] = results_df['price_sentiment_polarity'].apply(
                lambda x: self.get_sentiment_label(x, 'textblob')
            )
        
        # Process timestamp if provided
        if timestamp_column and timestamp_column in results_df.columns:
            results_df[timestamp_column] = pd.to_datetime(results_df[timestamp_column])
            results_df = results_df.sort_values(timestamp_column)
        
        self.results = results_df
        return results_df
    
    def get_summary_statistics(self, df=None):
        """
        Generate summary statistics for sentiment analysis.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze. Uses self.results if None.
            
        Returns:
            dict: Dictionary with summary statistics
        """
        if df is None:
            df = self.results
            
        if df is None:
            raise ValueError("No data to analyze. Run analyze_dataframe first.")
        
        stats = {
            'total_responses': len(df),
            'textblob_positive_percent': (df['textblob_sentiment'] == 'Positive').mean() * 100,
            'textblob_negative_percent': (df['textblob_sentiment'] == 'Negative').mean() * 100,
            'textblob_neutral_percent': (df['textblob_sentiment'] == 'Neutral').mean() * 100,
            'vader_positive_percent': (df['vader_sentiment'] == 'Positive').mean() * 100,
            'vader_negative_percent': (df['vader_sentiment'] == 'Negative').mean() * 100,
            'vader_neutral_percent': (df['vader_sentiment'] == 'Neutral').mean() * 100,
            'avg_textblob_polarity': df['textblob_polarity'].mean(),
            'avg_vader_compound': df['vader_compound'].mean(),
            'method_agreement': (df['textblob_sentiment'] == df['vader_sentiment']).mean() * 100
        }
        
        # Add rating correlation if available
        if 'rating' in df.columns:
            stats['textblob_rating_correlation'] = df['textblob_polarity'].corr(df['rating'])
            stats['vader_rating_correlation'] = df['vader_compound'].corr(df['rating'])
            stats['avg_rating'] = df['rating'].mean()
        
        return stats
    
    def create_visualizations(self, df=None, save_plots=False, plot_dir='plots/'):
        """
        Create comprehensive visualizations for sentiment analysis.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to visualize. Uses self.results if None.
            save_plots (bool): Whether to save plots to files
            plot_dir (str): Directory to save plots
        """
        if df is None:
            df = self.results
            
        if df is None:
            raise ValueError("No data to visualize. Run analyze_dataframe first.")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Sentiment Distribution (TextBlob)
        sentiment_counts = df['textblob_sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Sentiment Distribution (TextBlob)')
        
        # Plot 2: Sentiment vs Rating (if available)
        if 'rating' in df.columns:
            sns.boxplot(data=df, x='textblob_sentiment', y='rating', ax=axes[0, 1])
            axes[0, 1].set_title('Rating Distribution by Sentiment')
            axes[0, 1].set_ylabel('Rating')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Rating Data\nAvailable', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Rating vs Sentiment (No Data)')
        
        # Plot 3: Method Comparison
        axes[0, 2].scatter(df['textblob_polarity'], df['vader_compound'], alpha=0.7)
        axes[0, 2].set_xlabel('TextBlob Polarity')
        axes[0, 2].set_ylabel('VADER Compound')
        axes[0, 2].set_title('TextBlob vs VADER Comparison')
        
        # Add correlation line
        correlation = df['textblob_polarity'].corr(df['vader_compound'])
        axes[0, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 2].transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", alpha=0.8))
        
        # Plot 4: Sentiment Over Time (if timestamp available)
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col and pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df_sorted = df.sort_values(timestamp_col)
            axes[1, 0].plot(df_sorted[timestamp_col], df_sorted['textblob_polarity'], 
                           marker='o', linestyle='-', alpha=0.7)
            axes[1, 0].set_title('Sentiment Trend Over Time')
            axes[1, 0].set_ylabel('Sentiment Polarity')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Timestamp Data\nAvailable', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Sentiment Over Time (No Data)')
        
        # Plot 5: Sentiment Intensity Distribution
        axes[1, 1].hist(df['textblob_polarity'], bins=20, alpha=0.7, color='skyblue', 
                       edgecolor='black')
        axes[1, 1].axvline(df['textblob_polarity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["textblob_polarity"].mean():.3f}')
        axes[1, 1].set_xlabel('Sentiment Polarity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Sentiment Polarity Distribution')
        axes[1, 1].legend()
        
        # Plot 6: Top Positive and Negative Words
        all_text = ' '.join(df.iloc[:, 0].astype(str))  # Assuming first text column
        words = all_text.lower().split()
        word_sentiments = [(word, TextBlob(word).sentiment.polarity) for word in set(words) 
                          if len(word) > 3]
        word_sentiments.sort(key=lambda x: x[1])
        
        # Get top 5 negative and positive words
        top_negative = word_sentiments[:5]
        top_positive = word_sentiments[-5:]
        
        words_data = top_negative + top_positive
        word_names = [w[0] for w in words_data]
        word_scores = [w[1] for w in words_data]
        
        colors = ['red' if score < 0 else 'green' for score in word_scores]
        axes[1, 2].barh(word_names, word_scores, color=colors, alpha=0.7)
        axes[1, 2].set_xlabel('Sentiment Polarity')
        axes[1, 2].set_title('Top Positive/Negative Words')
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f'{plot_dir}/sentiment_dashboard.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, df=None, filename='sentiment_analysis_results.csv'):
        """
        Export sentiment analysis results to CSV.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to export. Uses self.results if None.
            filename (str): Output filename
        """
        if df is None:
            df = self.results
            
        if df is None:
            raise ValueError("No data to export. Run analyze_dataframe first.")
        
        df.to_csv(filename, index=False)
        print(f"Results exported to: {filename}")
    
    def get_insights(self, df=None):
        """
        Generate business insights from sentiment analysis.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze. Uses self.results if None.
            
        Returns:
            dict: Dictionary with business insights
        """
        if df is None:
            df = self.results
            
        if df is None:
            raise ValueError("No data to analyze. Run analyze_dataframe first.")
        
        stats = self.get_summary_statistics(df)
        insights = {
            'overall_sentiment': 'Positive' if stats['avg_textblob_polarity'] > 0.1 else 
                               'Negative' if stats['avg_textblob_polarity'] < -0.1 else 'Neutral',
            'dominant_sentiment': df['textblob_sentiment'].mode()[0],
            'sentiment_consistency': 'High' if stats['method_agreement'] > 80 else 
                                   'Medium' if stats['method_agreement'] > 60 else 'Low',
            'recommendations': []
        }
        
        # Generate recommendations
        if stats['textblob_positive_percent'] > 70:
            insights['recommendations'].append("Strong positive sentiment - leverage for marketing")
        if stats['textblob_negative_percent'] > 20:
            insights['recommendations'].append("Address negative feedback to improve satisfaction")
        if 'textblob_rating_correlation' in stats and stats['textblob_rating_correlation'] > 0.5:
            insights['recommendations'].append("Sentiment strongly correlates with ratings - focus on improving sentiment")
        
        return insights


# Example usage and testing functions
def create_sample_data():
    """Create sample survey data for testing."""
    sample_data = {
        'timestamp': pd.date_range('2025-05-24', periods=10, freq='D'),
        'rating': [10, 9, 9, 8, 10, 9, 10, 7, 9, 10],
        'feedback': [
            'Amazing and innovative product!',
            'Very helpful and useful',
            'Convenient and well-designed',
            'Good but could be better',
            'Excellent service and support',
            'Love this product so much',
            'Fantastic experience overall',
            'Okay, nothing special',
            'Great value for money',
            'Outstanding quality and features'
        ],
        'price_feedback': [
            'Worth every penny',
            'Reasonably priced',
            'A bit expensive but okay',
            'Too expensive for what it offers',
            'Great value',
            'Would pay more for this',
            'Perfect price point',
            'Overpriced',
            'Good deal',
            'Excellent value proposition'
        ]
    }
    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Example usage
    print("Sentiment Analyzer - Example Usage")
    print("=" * 50)
    
    # Create sample data
    sample_df = create_sample_data()
    print("Sample data created:")
    print(sample_df.head())
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze the data
    results = analyzer.analyze_dataframe(
        sample_df, 
        text_column='feedback',
        rating_column='rating',
        timestamp_column='timestamp',
        price_column='price_feedback'
    )
    
    # Get summary statistics
    stats = analyzer.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get business insights
    insights = analyzer.get_insights()
    print("\nBusiness Insights:")
    print(f"Overall Sentiment: {insights['overall_sentiment']}")
    print(f"Dominant Sentiment: {insights['dominant_sentiment']}")
    print(f"Method Agreement: {insights['sentiment_consistency']}")
    print("Recommendations:")
    for rec in insights['recommendations']:
        print(f"- {rec}")
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Export results
    analyzer.export_results('sample_results.csv')
