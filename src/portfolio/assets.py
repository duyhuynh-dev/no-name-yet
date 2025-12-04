"""
Asset Class Definitions and Universe Management

Defines asset classes, individual assets, and manages
the tradeable universe.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd


class AssetClass(Enum):
    """Asset class classification."""
    EQUITY_US = "equity_us"
    EQUITY_INTL = "equity_intl"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FOREX = "forex"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"
    ETF = "etf"
    INDEX = "index"


class Sector(Enum):
    """Equity sectors."""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"
    OTHER = "other"


@dataclass
class Asset:
    """Individual tradeable asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    
    # Optional metadata
    sector: Optional[Sector] = None
    country: str = "US"
    currency: str = "USD"
    exchange: str = ""
    
    # Trading parameters
    min_trade_size: float = 1.0
    tick_size: float = 0.01
    margin_requirement: float = 1.0  # 1.0 = no margin
    
    # Risk parameters
    beta: float = 1.0
    volatility: float = 0.0
    avg_volume: float = 0.0
    
    # Status
    is_tradeable: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_class": self.asset_class.value,
            "sector": self.sector.value if self.sector else None,
            "country": self.country,
            "currency": self.currency,
            "beta": self.beta,
            "volatility": self.volatility,
            "is_tradeable": self.is_tradeable,
        }


@dataclass
class CryptoAsset(Asset):
    """Cryptocurrency-specific asset."""
    base_currency: str = ""
    quote_currency: str = "USDT"
    is_perpetual: bool = False
    funding_rate: float = 0.0
    
    def __post_init__(self):
        self.asset_class = AssetClass.CRYPTO
        self.margin_requirement = 0.1  # Typically 10x leverage available


@dataclass
class OptionsAsset(Asset):
    """Options contract."""
    underlying: str = ""
    strike: float = 0.0
    expiration: Optional[datetime] = None
    option_type: str = "call"  # call or put
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    def __post_init__(self):
        self.asset_class = AssetClass.OPTIONS
    
    @property
    def is_itm(self) -> bool:
        """Check if option is in the money (requires current price)."""
        return False  # Would need spot price
    
    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        if self.expiration:
            return (self.expiration - datetime.now()).days
        return 0


@dataclass
class FuturesAsset(Asset):
    """Futures contract."""
    underlying: str = ""
    expiration: Optional[datetime] = None
    contract_size: float = 1.0
    
    def __post_init__(self):
        self.asset_class = AssetClass.FUTURES


class AssetUniverse:
    """
    Manages the tradeable asset universe.
    
    Provides filtering, grouping, and universe management.
    """
    
    def __init__(self):
        """Initialize Asset Universe."""
        self._assets: Dict[str, Asset] = {}
        self._by_class: Dict[AssetClass, Set[str]] = {ac: set() for ac in AssetClass}
        self._by_sector: Dict[Sector, Set[str]] = {s: set() for s in Sector}
    
    def add_asset(self, asset: Asset) -> None:
        """Add asset to universe."""
        self._assets[asset.symbol] = asset
        self._by_class[asset.asset_class].add(asset.symbol)
        if asset.sector:
            self._by_sector[asset.sector].add(asset.symbol)
    
    def remove_asset(self, symbol: str) -> Optional[Asset]:
        """Remove asset from universe."""
        if symbol not in self._assets:
            return None
        
        asset = self._assets.pop(symbol)
        self._by_class[asset.asset_class].discard(symbol)
        if asset.sector:
            self._by_sector[asset.sector].discard(symbol)
        
        return asset
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """Get asset by symbol."""
        return self._assets.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in universe."""
        return list(self._assets.keys())
    
    def get_by_class(self, asset_class: AssetClass) -> List[Asset]:
        """Get assets by asset class."""
        return [self._assets[s] for s in self._by_class[asset_class]]
    
    def get_by_sector(self, sector: Sector) -> List[Asset]:
        """Get assets by sector."""
        return [self._assets[s] for s in self._by_sector[sector]]
    
    def get_tradeable(self) -> List[Asset]:
        """Get all tradeable assets."""
        return [a for a in self._assets.values() if a.is_tradeable]
    
    def filter_assets(
        self,
        asset_class: Optional[AssetClass] = None,
        sector: Optional[Sector] = None,
        min_volume: Optional[float] = None,
        max_volatility: Optional[float] = None,
        countries: Optional[List[str]] = None,
    ) -> List[Asset]:
        """
        Filter assets by multiple criteria.
        
        Args:
            asset_class: Filter by asset class
            sector: Filter by sector
            min_volume: Minimum average volume
            max_volatility: Maximum volatility
            countries: List of allowed countries
            
        Returns:
            List of matching assets
        """
        results = list(self._assets.values())
        
        if asset_class:
            results = [a for a in results if a.asset_class == asset_class]
        
        if sector:
            results = [a for a in results if a.sector == sector]
        
        if min_volume is not None:
            results = [a for a in results if a.avg_volume >= min_volume]
        
        if max_volatility is not None:
            results = [a for a in results if a.volatility <= max_volatility]
        
        if countries:
            results = [a for a in results if a.country in countries]
        
        return results
    
    def update_metrics(
        self,
        symbol: str,
        volatility: Optional[float] = None,
        beta: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> None:
        """Update asset risk metrics."""
        if symbol not in self._assets:
            return
        
        asset = self._assets[symbol]
        
        if volatility is not None:
            asset.volatility = volatility
        if beta is not None:
            asset.beta = beta
        if avg_volume is not None:
            asset.avg_volume = avg_volume
        
        asset.last_updated = datetime.now()
    
    def get_correlation_groups(
        self,
        returns_df: pd.DataFrame,
        threshold: float = 0.7,
    ) -> List[List[str]]:
        """
        Group highly correlated assets.
        
        Args:
            returns_df: DataFrame of asset returns
            threshold: Correlation threshold for grouping
            
        Returns:
            List of correlated asset groups
        """
        corr_matrix = returns_df.corr()
        
        groups = []
        used = set()
        
        for symbol in corr_matrix.columns:
            if symbol in used:
                continue
            
            group = [symbol]
            used.add(symbol)
            
            for other in corr_matrix.columns:
                if other in used:
                    continue
                if abs(corr_matrix.loc[symbol, other]) >= threshold:
                    group.append(other)
                    used.add(other)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def calculate_universe_stats(
        self,
        returns_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate universe-level statistics.
        
        Args:
            returns_df: DataFrame of asset returns
            
        Returns:
            Universe statistics
        """
        symbols = [s for s in returns_df.columns if s in self._assets]
        
        if not symbols:
            return {}
        
        returns = returns_df[symbols]
        
        return {
            "num_assets": len(symbols),
            "avg_return": returns.mean().mean(),
            "avg_volatility": returns.std().mean(),
            "avg_correlation": returns.corr().values[np.triu_indices_from(
                returns.corr().values, k=1
            )].mean(),
            "by_class": {
                ac.value: len(self._by_class[ac])
                for ac in AssetClass
                if len(self._by_class[ac]) > 0
            },
        }
    
    def create_default_universe(self) -> None:
        """Create a default asset universe with common assets."""
        # US Large Cap Equities
        large_caps = [
            ("AAPL", "Apple Inc", Sector.TECHNOLOGY),
            ("MSFT", "Microsoft Corp", Sector.TECHNOLOGY),
            ("GOOGL", "Alphabet Inc", Sector.COMMUNICATION),
            ("AMZN", "Amazon.com Inc", Sector.CONSUMER_DISCRETIONARY),
            ("NVDA", "NVIDIA Corp", Sector.TECHNOLOGY),
            ("META", "Meta Platforms", Sector.COMMUNICATION),
            ("TSLA", "Tesla Inc", Sector.CONSUMER_DISCRETIONARY),
            ("JPM", "JPMorgan Chase", Sector.FINANCIALS),
            ("V", "Visa Inc", Sector.FINANCIALS),
            ("JNJ", "Johnson & Johnson", Sector.HEALTHCARE),
        ]
        
        for symbol, name, sector in large_caps:
            self.add_asset(Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.EQUITY_US,
                sector=sector,
            ))
        
        # ETFs
        etfs = [
            ("SPY", "SPDR S&P 500 ETF"),
            ("QQQ", "Invesco QQQ Trust"),
            ("IWM", "iShares Russell 2000"),
            ("DIA", "SPDR Dow Jones"),
            ("VTI", "Vanguard Total Stock"),
            ("TLT", "iShares 20+ Year Treasury"),
            ("GLD", "SPDR Gold Shares"),
            ("XLF", "Financial Select Sector"),
            ("XLE", "Energy Select Sector"),
            ("XLK", "Technology Select Sector"),
        ]
        
        for symbol, name in etfs:
            self.add_asset(Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.ETF,
            ))
        
        # Crypto
        cryptos = [
            ("BTC", "Bitcoin"),
            ("ETH", "Ethereum"),
            ("SOL", "Solana"),
            ("BNB", "Binance Coin"),
            ("XRP", "Ripple"),
        ]
        
        for symbol, name in cryptos:
            self.add_asset(CryptoAsset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.CRYPTO,
                base_currency=symbol,
            ))
        
        # Forex
        forex_pairs = [
            ("EURUSD", "Euro/US Dollar"),
            ("GBPUSD", "British Pound/US Dollar"),
            ("USDJPY", "US Dollar/Japanese Yen"),
            ("AUDUSD", "Australian Dollar/US Dollar"),
        ]
        
        for symbol, name in forex_pairs:
            self.add_asset(Asset(
                symbol=symbol,
                name=name,
                asset_class=AssetClass.FOREX,
                margin_requirement=0.02,  # 50x leverage typical
            ))
    
    def __len__(self) -> int:
        return len(self._assets)
    
    def __contains__(self, symbol: str) -> bool:
        return symbol in self._assets

