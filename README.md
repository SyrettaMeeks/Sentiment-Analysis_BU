# Sentiment Analysis for Business Understanding (BU)

A comprehensive Python toolkit for performing sentiment analysis on survey data and customer feedback to drive business insights.

## 🎯 Project Overview

This repository contains tools and tutorials for analyzing customer sentiment from survey responses, with a focus on understanding business impact through correlation with ratings, pricing feedback, and customer satisfaction metrics.

## 📊 Features

- **Multiple Sentiment Analysis Methods**: TextBlob and VADER implementations
- **Business Correlation Analysis**: Connect sentiment with ratings and pricing willingness
- **Comprehensive Visualizations**: Charts and graphs for data presentation
- **Export Capabilities**: Save results to CSV for further analysis
- **Tutorial-Based Learning**: Step-by-step guide from beginner to advanced

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas textblob vaderSentiment matplotlib seaborn
```

### Basic Usage

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze your data
results = analyzer.analyze_survey_data('your_data.csv')

# Generate visualizations
analyzer.create_visualizations(results)

# Export results
analyzer.export_results(results, 'output.csv')
```

## 📁 Repository Structure

```
Sentiment-Analysis_BU/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── sentiment_analyzer.py     # Main analysis class
├── tutorial.py              # Complete tutorial script
├── examples/                 # Example datasets and outputs
│   ├── sample_survey_data.csv
│   └── example_output.csv
├── docs/                    # Documentation
│   ├── getting_started.md
│   ├── api_reference.md
│   └── business_insights.md
└── tests/                   # Unit tests
    └── test_sentiment.py
```

## 📈 Business Applications

### Customer Feedback Analysis
- Analyze customer reviews and feedback
- Identify sentiment trends over time
- Correlate sentiment with business metrics

### Survey Analysis
- Process survey responses for sentiment
- Understand customer satisfaction drivers
- Price sensitivity analysis through sentiment

### Market Research
- Competitive sentiment analysis
- Product feedback evaluation
- Feature request prioritization

## 🔧 Technical Details

### Sentiment Analysis Methods

**TextBlob**
- Good for: General sentiment analysis
- Outputs: Polarity (-1 to 1) and Subjectivity (0 to 1)
- Best for: Clean, formal text

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Good for: Social media text, informal language
- Outputs: Compound score and individual pos/neg/neu scores
- Best for: Text with emoticons, slang, intensifiers

### Key Metrics

- **Sentiment-Rating Correlation**: How sentiment aligns with numerical ratings
- **Price Sensitivity by Sentiment**: Willingness to pay based on emotional response
- **Temporal Sentiment Analysis**: Sentiment trends over time

## 📊 Sample Analysis Results

From our example survey data:
- **Positive Sentiment**: 75% of responses
- **Average Rating**: 9.2/10
- **Sentiment-Rating Correlation**: 0.68
- **Price Sensitivity**: Positive sentiment correlates with higher willingness to pay

## 🎓 Learning Path

1. **Start Here**: Run `tutorial.py` for complete walkthrough
2. **Understand Methods**: Read `docs/api_reference.md`
3. **Business Applications**: Review `docs/business_insights.md`
4. **Advanced Usage**: Explore `sentiment_analyzer.py`

## 📋 Examples

### Basic Sentiment Analysis
```python
from textblob import TextBlob

text = "This product is amazing and innovative!"
blob = TextBlob(text)
print(f"Sentiment: {blob.sentiment.polarity}")  # Output: 0.625 (positive)
```

### Survey Data Analysis
```python
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer

# Load your survey data
df = pd.read_csv('survey_responses.csv')

# Analyze sentiment
analyzer = SentimentAnalyzer()
results = analyzer.analyze_dataframe(df, text_column='feedback')

# View results
print(results.head())
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Documentation**: Check the `docs/` folder
- **Issues**: Open an issue on GitHub
- **Examples**: See the `examples/` folder

## 🏗️ Roadmap

- [ ] Add more sentiment analysis models (RoBERTa, BERT)
- [ ] Implement real-time sentiment monitoring
- [ ] Add multilingual support
- [ ] Create web dashboard interface
- [ ] Integration with popular survey platforms

## 📚 References

- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Built for Business Understanding** | **Python 3.7+** | **MIT License**
