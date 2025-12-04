"""
Sentiment Analysis for Financial Markets

Provides LLM-based sentiment analysis for news, social media,
and earnings reports.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


class SentimentLabel(Enum):
    """Sentiment classification."""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    sentiment: SentimentLabel
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def numeric_sentiment(self) -> float:
        """Get sentiment as a float between -1 and 1."""
        return self.sentiment.value / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "sentiment": self.sentiment.name,
            "confidence": self.confidence,
            "numeric_sentiment": self.numeric_sentiment,
            "scores": self.scores,
            "keywords": self.keywords,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NewsItem:
    """News article data."""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None


@dataclass
class SocialPost:
    """Social media post data."""
    content: str
    platform: str
    author: str
    posted_at: datetime
    likes: int = 0
    shares: int = 0
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""
    
    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        pass
    
    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        pass


class RuleBasedSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Rule-based sentiment analyzer.
    
    Uses keyword matching and financial lexicon for
    quick sentiment classification without ML models.
    """
    
    # Financial sentiment lexicon
    BULLISH_WORDS = {
        "strong": 1, "bullish": 2, "surge": 2, "rally": 2, "gain": 1,
        "growth": 1, "profit": 1, "beat": 1, "exceed": 1, "outperform": 2,
        "upgrade": 2, "buy": 1, "positive": 1, "optimistic": 1, "robust": 1,
        "momentum": 1, "breakout": 2, "upside": 1, "record": 1, "high": 1,
        "boom": 2, "soar": 2, "jump": 1, "climb": 1, "advance": 1,
        "recovery": 1, "opportunity": 1, "confidence": 1, "strong": 1,
    }
    
    BEARISH_WORDS = {
        "weak": -1, "bearish": -2, "crash": -2, "plunge": -2, "loss": -1,
        "decline": -1, "miss": -1, "below": -1, "underperform": -2, "downgrade": -2,
        "sell": -1, "negative": -1, "pessimistic": -1, "concern": -1, "risk": -1,
        "breakdown": -2, "downside": -1, "low": -1, "bust": -2, "sink": -2,
        "drop": -1, "fall": -1, "retreat": -1, "recession": -2, "fear": -1,
        "volatility": -1, "uncertainty": -1, "warning": -1, "cut": -1,
    }
    
    INTENSIFIERS = {"very", "extremely", "highly", "significantly", "strongly"}
    NEGATORS = {"not", "no", "never", "neither", "without", "barely", "hardly"}
    
    def __init__(self):
        """Initialize the rule-based analyzer."""
        self.all_words = {**self.BULLISH_WORDS, **self.BEARISH_WORDS}
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with classification
        """
        # Preprocess
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Score calculation
        score = 0.0
        matched_keywords = []
        intensifier_count = 0
        negation_active = False
        
        for i, word in enumerate(words):
            # Check for negators
            if word in self.NEGATORS:
                negation_active = True
                continue
            
            # Check for intensifiers
            if word in self.INTENSIFIERS:
                intensifier_count += 1
                continue
            
            # Check for sentiment words
            if word in self.all_words:
                word_score = self.all_words[word]
                
                # Apply negation
                if negation_active:
                    word_score *= -0.5
                    negation_active = False
                
                # Apply intensifiers
                if intensifier_count > 0:
                    word_score *= (1 + 0.5 * intensifier_count)
                    intensifier_count = 0
                
                score += word_score
                matched_keywords.append(word)
        
        # Normalize score
        if matched_keywords:
            normalized_score = score / (len(matched_keywords) * 2)
        else:
            normalized_score = 0.0
        
        # Clamp to [-1, 1]
        normalized_score = max(-1.0, min(1.0, normalized_score))
        
        # Convert to label
        if normalized_score >= 0.5:
            sentiment = SentimentLabel.VERY_BULLISH
        elif normalized_score >= 0.2:
            sentiment = SentimentLabel.BULLISH
        elif normalized_score <= -0.5:
            sentiment = SentimentLabel.VERY_BEARISH
        elif normalized_score <= -0.2:
            sentiment = SentimentLabel.BEARISH
        else:
            sentiment = SentimentLabel.NEUTRAL
        
        # Confidence based on keyword coverage
        word_coverage = len(matched_keywords) / max(len(words), 1)
        confidence = min(1.0, word_coverage * 3 + 0.3)
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            scores={
                "raw_score": score,
                "normalized_score": normalized_score,
            },
            keywords=matched_keywords,
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]


class SentimentAnalyzer:
    """
    Main sentiment analyzer with multiple backends.
    
    Supports rule-based and (future) LLM-based analysis.
    """
    
    def __init__(
        self,
        backend: str = "rule_based",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            backend: Analysis backend ("rule_based", "openai", "local_llm")
            api_key: API key for external services
        """
        self.backend = backend
        self.api_key = api_key
        
        if backend == "rule_based":
            self._analyzer = RuleBasedSentimentAnalyzer()
        elif backend == "openai":
            self._analyzer = self._create_openai_analyzer()
        else:
            self._analyzer = RuleBasedSentimentAnalyzer()
    
    def _create_openai_analyzer(self) -> BaseSentimentAnalyzer:
        """Create OpenAI-based analyzer (stub for future implementation)."""
        # Would require openai package and API key
        # For now, fall back to rule-based
        return RuleBasedSentimentAnalyzer()
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        return self._analyzer.analyze(text)
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return self._analyzer.analyze_batch(texts)
    
    def analyze_news(self, news: NewsItem) -> NewsItem:
        """Analyze sentiment of a news item."""
        combined_text = f"{news.title}. {news.content}"
        news.sentiment = self.analyze(combined_text)
        return news
    
    def analyze_social(self, post: SocialPost) -> SocialPost:
        """Analyze sentiment of a social media post."""
        post.sentiment = self.analyze(post.content)
        return post
    
    def get_aggregate_sentiment(
        self,
        results: List[SentimentResult],
        weight_by_confidence: bool = True,
    ) -> Tuple[SentimentLabel, float]:
        """
        Calculate aggregate sentiment from multiple results.
        
        Args:
            results: List of sentiment results
            weight_by_confidence: Whether to weight by confidence
            
        Returns:
            Tuple of (aggregate_sentiment, aggregate_score)
        """
        if not results:
            return SentimentLabel.NEUTRAL, 0.0
        
        if weight_by_confidence:
            total_weight = sum(r.confidence for r in results)
            if total_weight == 0:
                avg_score = 0.0
            else:
                avg_score = sum(
                    r.numeric_sentiment * r.confidence for r in results
                ) / total_weight
        else:
            avg_score = sum(r.numeric_sentiment for r in results) / len(results)
        
        # Convert to label
        if avg_score >= 0.3:
            sentiment = SentimentLabel.VERY_BULLISH
        elif avg_score >= 0.1:
            sentiment = SentimentLabel.BULLISH
        elif avg_score <= -0.3:
            sentiment = SentimentLabel.VERY_BEARISH
        elif avg_score <= -0.1:
            sentiment = SentimentLabel.BEARISH
        else:
            sentiment = SentimentLabel.NEUTRAL
        
        return sentiment, avg_score


class NewsProcessor:
    """
    Process and analyze financial news.
    
    Handles news aggregation, filtering, and sentiment analysis.
    """
    
    def __init__(
        self,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ):
        """
        Initialize News Processor.
        
        Args:
            sentiment_analyzer: Sentiment analyzer instance
        """
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self._news_cache: List[NewsItem] = []
    
    def extract_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text.
        
        Args:
            text: Text to search
            
        Returns:
            List of potential stock symbols
        """
        # Match $SYMBOL or SYMBOL patterns
        pattern = r'\$?([A-Z]{1,5})(?:\s|$|[.,!?])'
        matches = re.findall(pattern, text.upper())
        
        # Filter common words that look like symbols
        common_words = {"I", "A", "THE", "AND", "OR", "FOR", "TO", "IN", "ON", "AT"}
        symbols = [m for m in matches if m not in common_words and len(m) >= 2]
        
        return list(set(symbols))
    
    def process_news(
        self,
        title: str,
        content: str,
        source: str,
        url: str,
        published_at: Optional[datetime] = None,
    ) -> NewsItem:
        """
        Process a news article.
        
        Args:
            title: Article title
            content: Article content
            source: News source
            url: Article URL
            published_at: Publication datetime
            
        Returns:
            Processed NewsItem with sentiment
        """
        published_at = published_at or datetime.now()
        
        # Extract symbols
        symbols = self.extract_symbols(f"{title} {content}")
        
        # Create news item
        news = NewsItem(
            title=title,
            content=content,
            source=source,
            url=url,
            published_at=published_at,
            symbols=symbols,
        )
        
        # Analyze sentiment
        news = self.sentiment_analyzer.analyze_news(news)
        
        # Cache
        self._news_cache.append(news)
        
        return news
    
    def get_symbol_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24,
    ) -> Tuple[SentimentLabel, float, int]:
        """
        Get aggregate sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours to look back
            
        Returns:
            Tuple of (sentiment, score, news_count)
        """
        cutoff = datetime.now().timestamp() - (lookback_hours * 3600)
        
        relevant_news = [
            n for n in self._news_cache
            if symbol.upper() in n.symbols
            and n.published_at.timestamp() > cutoff
            and n.sentiment is not None
        ]
        
        if not relevant_news:
            return SentimentLabel.NEUTRAL, 0.0, 0
        
        sentiments = [n.sentiment for n in relevant_news]
        aggregate_sentiment, score = self.sentiment_analyzer.get_aggregate_sentiment(
            sentiments
        )
        
        return aggregate_sentiment, score, len(relevant_news)
    
    def get_recent_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[NewsItem]:
        """Get recent news, optionally filtered by symbol."""
        news = self._news_cache
        
        if symbol:
            news = [n for n in news if symbol.upper() in n.symbols]
        
        # Sort by date, newest first
        news = sorted(news, key=lambda n: n.published_at, reverse=True)
        
        return news[:limit]


class SentimentSignalGenerator:
    """
    Generate trading signals from sentiment analysis.
    
    Combines sentiment from multiple sources into actionable signals.
    """
    
    def __init__(
        self,
        news_weight: float = 0.6,
        social_weight: float = 0.4,
        threshold_bullish: float = 0.3,
        threshold_bearish: float = -0.3,
    ):
        """
        Initialize Signal Generator.
        
        Args:
            news_weight: Weight for news sentiment
            social_weight: Weight for social media sentiment
            threshold_bullish: Threshold for bullish signal
            threshold_bearish: Threshold for bearish signal
        """
        self.news_weight = news_weight
        self.social_weight = social_weight
        self.threshold_bullish = threshold_bullish
        self.threshold_bearish = threshold_bearish
    
    def generate_signal(
        self,
        news_sentiment: float,
        social_sentiment: float,
        news_count: int,
        social_count: int,
    ) -> Dict[str, Any]:
        """
        Generate trading signal from sentiment.
        
        Args:
            news_sentiment: News sentiment score [-1, 1]
            social_sentiment: Social sentiment score [-1, 1]
            news_count: Number of news items
            social_count: Number of social posts
            
        Returns:
            Signal dictionary with action and confidence
        """
        # Adjust weights based on data availability
        total_items = news_count + social_count
        if total_items == 0:
            return {
                "action": "hold",
                "confidence": 0.0,
                "combined_sentiment": 0.0,
                "reason": "No sentiment data available",
            }
        
        # Calculate adjusted weights
        if news_count == 0:
            adj_news_weight = 0
            adj_social_weight = 1
        elif social_count == 0:
            adj_news_weight = 1
            adj_social_weight = 0
        else:
            adj_news_weight = self.news_weight
            adj_social_weight = self.social_weight
        
        # Normalize weights
        total_weight = adj_news_weight + adj_social_weight
        adj_news_weight /= total_weight
        adj_social_weight /= total_weight
        
        # Combined sentiment
        combined = (
            news_sentiment * adj_news_weight +
            social_sentiment * adj_social_weight
        )
        
        # Determine action
        if combined >= self.threshold_bullish:
            action = "buy"
            confidence = min(1.0, (combined - self.threshold_bullish) / 0.7 + 0.3)
        elif combined <= self.threshold_bearish:
            action = "sell"
            confidence = min(1.0, (self.threshold_bearish - combined) / 0.7 + 0.3)
        else:
            action = "hold"
            confidence = 1.0 - abs(combined) / max(
                abs(self.threshold_bullish), abs(self.threshold_bearish)
            )
        
        # Adjust confidence by data volume
        volume_factor = min(1.0, total_items / 10)
        confidence *= (0.5 + 0.5 * volume_factor)
        
        return {
            "action": action,
            "confidence": confidence,
            "combined_sentiment": combined,
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "news_count": news_count,
            "social_count": social_count,
        }

