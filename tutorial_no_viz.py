# Stop the current process
Ctrl + C

# Then run just the analyzer without full tutorial
python -c "
from sentiment_analyzer import SentimentAnalyzer
import pandas as pd
print('Sentiment Analysis Complete!')
print('Check your results in sentiment_analysis_results.csv')
"
