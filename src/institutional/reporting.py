"""
Reporting and Analytics System

Generates various reports for performance analysis,
compliance, and stakeholder communication.
"""

import json
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np


class ReportType(Enum):
    """Types of reports."""
    DAILY_PNL = "daily_pnl"
    WEEKLY_PNL = "weekly_pnl"
    MONTHLY_PNL = "monthly_pnl"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    RISK_REPORT = "risk_report"
    TRADE_ACTIVITY = "trade_activity"
    POSITION_SUMMARY = "position_summary"
    FACTOR_EXPOSURE = "factor_exposure"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    TAX_LOT = "tax_lot"
    COMPLIANCE_SUMMARY = "compliance_summary"
    EXECUTIVE_SUMMARY = "executive_summary"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"


@dataclass
class Report:
    """Generated report."""
    report_id: str
    report_type: ReportType
    title: str
    
    # Time range
    start_date: date
    end_date: date
    
    # Content
    data: Dict[str, Any] = field(default_factory=dict)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: Optional[str] = None
    
    format: ReportFormat = ReportFormat.JSON
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "title": self.title,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "data": self.data,
            "sections": self.sections,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def to_html(self) -> str:
        """Generate HTML representation."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f4f4f4; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{self.title}</h1>
            <p>Period: {self.start_date} to {self.end_date}</p>
            <p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for section in self.sections:
            html += f"<h2>{section.get('title', 'Section')}</h2>"
            
            if 'table' in section:
                html += "<table>"
                table = section['table']
                if 'headers' in table:
                    html += "<tr>"
                    for h in table['headers']:
                        html += f"<th>{h}</th>"
                    html += "</tr>"
                if 'rows' in table:
                    for row in table['rows']:
                        html += "<tr>"
                        for cell in row:
                            html += f"<td>{cell}</td>"
                        html += "</tr>"
                html += "</table>"
            
            if 'text' in section:
                html += f"<p>{section['text']}</p>"
            
            if 'metrics' in section:
                html += "<div class='summary'>"
                for key, value in section['metrics'].items():
                    css_class = ""
                    if isinstance(value, (int, float)):
                        css_class = "positive" if value > 0 else "negative" if value < 0 else ""
                    html += f"<p><strong>{key}:</strong> <span class='{css_class}'>{value}</span></p>"
                html += "</div>"
        
        html += "</body></html>"
        return html


class ReportGenerator:
    """
    Report Generator.
    
    Creates various financial and compliance reports.
    """
    
    def __init__(self):
        """Initialize Report Generator."""
        self._reports: List[Report] = []
        self._report_counter = 0
    
    def _generate_report_id(self, report_type: ReportType) -> str:
        """Generate unique report ID."""
        self._report_counter += 1
        prefix = report_type.value[:3].upper()
        return f"RPT-{prefix}-{datetime.now().strftime('%Y%m%d')}-{self._report_counter:04d}"
    
    def generate_daily_pnl(
        self,
        trades: List[Dict[str, Any]],
        positions: Dict[str, Dict[str, float]],
        prices: Dict[str, float],
        report_date: Optional[date] = None,
        user_id: Optional[str] = None,
    ) -> Report:
        """
        Generate daily P&L report.
        
        Args:
            trades: List of trades for the day
            positions: Current positions with cost basis
            prices: Current prices
            report_date: Report date (default today)
            user_id: User generating report
            
        Returns:
            Generated report
        """
        report_date = report_date or date.today()
        
        # Calculate trade P&L
        trade_pnl = 0.0
        for trade in trades:
            if trade.get('side') == 'sell':
                trade_pnl += (trade['price'] - trade.get('cost_basis', trade['price'])) * trade['quantity']
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        position_details = []
        
        for symbol, pos in positions.items():
            current_price = prices.get(symbol, 0)
            cost_basis = pos.get('cost_basis', current_price)
            quantity = pos.get('quantity', 0)
            
            unrealized = (current_price - cost_basis) * quantity
            unrealized_pnl += unrealized
            
            position_details.append({
                'symbol': symbol,
                'quantity': quantity,
                'cost_basis': cost_basis,
                'current_price': current_price,
                'unrealized_pnl': unrealized,
                'unrealized_pnl_pct': (unrealized / (cost_basis * quantity) * 100) if cost_basis * quantity > 0 else 0,
            })
        
        total_pnl = trade_pnl + unrealized_pnl
        
        report = Report(
            report_id=self._generate_report_id(ReportType.DAILY_PNL),
            report_type=ReportType.DAILY_PNL,
            title=f"Daily P&L Report - {report_date}",
            start_date=report_date,
            end_date=report_date,
            generated_by=user_id,
            data={
                "trade_pnl": trade_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "num_trades": len(trades),
                "num_positions": len(positions),
            },
            sections=[
                {
                    "title": "P&L Summary",
                    "metrics": {
                        "Realized P&L": f"${trade_pnl:,.2f}",
                        "Unrealized P&L": f"${unrealized_pnl:,.2f}",
                        "Total P&L": f"${total_pnl:,.2f}",
                        "Number of Trades": len(trades),
                    },
                },
                {
                    "title": "Position Details",
                    "table": {
                        "headers": ["Symbol", "Qty", "Cost Basis", "Current Price", "Unrealized P&L", "P&L %"],
                        "rows": [
                            [
                                p['symbol'],
                                f"{p['quantity']:.2f}",
                                f"${p['cost_basis']:.2f}",
                                f"${p['current_price']:.2f}",
                                f"${p['unrealized_pnl']:,.2f}",
                                f"{p['unrealized_pnl_pct']:.2f}%",
                            ]
                            for p in position_details
                        ],
                    },
                },
            ],
        )
        
        self._reports.append(report)
        return report
    
    def generate_performance_attribution(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        benchmark_returns: pd.Series,
        start_date: date,
        end_date: date,
        user_id: Optional[str] = None,
    ) -> Report:
        """
        Generate performance attribution report.
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            benchmark_returns: Benchmark returns
            start_date: Start date
            end_date: End date
            user_id: User generating report
            
        Returns:
            Generated report
        """
        # Calculate portfolio returns
        symbols = list(weights.keys())
        available = [s for s in symbols if s in returns.columns]
        
        w = np.array([weights[s] for s in available])
        port_returns = (returns[available] * w).sum(axis=1)
        
        # Total returns
        port_total = (1 + port_returns).prod() - 1
        bench_total = (1 + benchmark_returns).prod() - 1
        
        # Active return
        active_return = port_total - bench_total
        
        # Contribution by asset
        contributions = []
        for symbol in available:
            asset_return = (1 + returns[symbol]).prod() - 1
            contribution = asset_return * weights[symbol]
            contributions.append({
                'symbol': symbol,
                'weight': weights[symbol],
                'return': asset_return,
                'contribution': contribution,
            })
        
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Risk metrics
        port_vol = port_returns.std() * np.sqrt(252)
        bench_vol = benchmark_returns.std() * np.sqrt(252)
        tracking_error = (port_returns - benchmark_returns).std() * np.sqrt(252)
        
        # Sharpe and Information ratios
        rf = 0.04 / 252  # Daily risk-free rate
        sharpe = (port_returns.mean() - rf) / port_returns.std() * np.sqrt(252) if port_returns.std() > 0 else 0
        info_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        report = Report(
            report_id=self._generate_report_id(ReportType.PERFORMANCE_ATTRIBUTION),
            report_type=ReportType.PERFORMANCE_ATTRIBUTION,
            title=f"Performance Attribution Report",
            start_date=start_date,
            end_date=end_date,
            generated_by=user_id,
            data={
                "portfolio_return": port_total,
                "benchmark_return": bench_total,
                "active_return": active_return,
                "portfolio_volatility": port_vol,
                "tracking_error": tracking_error,
                "sharpe_ratio": sharpe,
                "information_ratio": info_ratio,
            },
            sections=[
                {
                    "title": "Performance Summary",
                    "metrics": {
                        "Portfolio Return": f"{port_total * 100:.2f}%",
                        "Benchmark Return": f"{bench_total * 100:.2f}%",
                        "Active Return": f"{active_return * 100:.2f}%",
                        "Portfolio Volatility": f"{port_vol * 100:.2f}%",
                        "Tracking Error": f"{tracking_error * 100:.2f}%",
                        "Sharpe Ratio": f"{sharpe:.2f}",
                        "Information Ratio": f"{info_ratio:.2f}",
                    },
                },
                {
                    "title": "Return Attribution by Asset",
                    "table": {
                        "headers": ["Asset", "Weight", "Return", "Contribution"],
                        "rows": [
                            [
                                c['symbol'],
                                f"{c['weight'] * 100:.1f}%",
                                f"{c['return'] * 100:.2f}%",
                                f"{c['contribution'] * 100:.2f}%",
                            ]
                            for c in contributions[:10]
                        ],
                    },
                },
            ],
        )
        
        self._reports.append(report)
        return report
    
    def generate_risk_report(
        self,
        var_95: float,
        var_99: float,
        cvar: float,
        beta: float,
        max_drawdown: float,
        volatility: float,
        positions: Dict[str, float],
        report_date: Optional[date] = None,
        user_id: Optional[str] = None,
    ) -> Report:
        """
        Generate risk report.
        
        Args:
            var_95: 95% VaR
            var_99: 99% VaR
            cvar: Conditional VaR
            beta: Portfolio beta
            max_drawdown: Maximum drawdown
            volatility: Portfolio volatility
            positions: Current positions
            report_date: Report date
            user_id: User generating report
            
        Returns:
            Generated report
        """
        report_date = report_date or date.today()
        
        # Position concentration
        total_value = sum(abs(v) for v in positions.values())
        concentrations = [
            {'symbol': s, 'value': v, 'weight': abs(v) / total_value if total_value > 0 else 0}
            for s, v in positions.items()
        ]
        concentrations.sort(key=lambda x: x['weight'], reverse=True)
        
        # Top concentration
        top_5_concentration = sum(c['weight'] for c in concentrations[:5])
        
        report = Report(
            report_id=self._generate_report_id(ReportType.RISK_REPORT),
            report_type=ReportType.RISK_REPORT,
            title=f"Risk Report - {report_date}",
            start_date=report_date,
            end_date=report_date,
            generated_by=user_id,
            data={
                "var_95": var_95,
                "var_99": var_99,
                "cvar": cvar,
                "beta": beta,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "top_5_concentration": top_5_concentration,
            },
            sections=[
                {
                    "title": "Value at Risk",
                    "metrics": {
                        "VaR (95%)": f"${var_95:,.2f}",
                        "VaR (99%)": f"${var_99:,.2f}",
                        "CVaR / Expected Shortfall": f"${cvar:,.2f}",
                    },
                },
                {
                    "title": "Portfolio Risk Metrics",
                    "metrics": {
                        "Portfolio Beta": f"{beta:.2f}",
                        "Annual Volatility": f"{volatility * 100:.2f}%",
                        "Maximum Drawdown": f"{max_drawdown * 100:.2f}%",
                    },
                },
                {
                    "title": "Position Concentration",
                    "metrics": {
                        "Top 5 Concentration": f"{top_5_concentration * 100:.1f}%",
                        "Number of Positions": len(positions),
                    },
                    "table": {
                        "headers": ["Symbol", "Value", "Weight"],
                        "rows": [
                            [c['symbol'], f"${c['value']:,.2f}", f"{c['weight'] * 100:.1f}%"]
                            for c in concentrations[:10]
                        ],
                    },
                },
            ],
        )
        
        self._reports.append(report)
        return report
    
    def generate_executive_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        mtd_pnl: float,
        ytd_pnl: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        num_trades: int,
        report_date: Optional[date] = None,
        user_id: Optional[str] = None,
    ) -> Report:
        """Generate executive summary report."""
        report_date = report_date or date.today()
        
        report = Report(
            report_id=self._generate_report_id(ReportType.EXECUTIVE_SUMMARY),
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title=f"Executive Summary - {report_date}",
            start_date=report_date,
            end_date=report_date,
            generated_by=user_id,
            data={
                "portfolio_value": portfolio_value,
                "daily_pnl": daily_pnl,
                "mtd_pnl": mtd_pnl,
                "ytd_pnl": ytd_pnl,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
            },
            sections=[
                {
                    "title": "Portfolio Overview",
                    "metrics": {
                        "Total Portfolio Value": f"${portfolio_value:,.2f}",
                        "Daily P&L": f"${daily_pnl:,.2f}",
                        "MTD P&L": f"${mtd_pnl:,.2f}",
                        "YTD P&L": f"${ytd_pnl:,.2f}",
                    },
                },
                {
                    "title": "Performance Metrics",
                    "metrics": {
                        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
                        "Maximum Drawdown": f"{max_drawdown * 100:.2f}%",
                        "Win Rate": f"{win_rate * 100:.1f}%",
                        "Total Trades": num_trades,
                    },
                },
            ],
        )
        
        self._reports.append(report)
        return report
    
    def get_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 50,
    ) -> List[Report]:
        """Get generated reports."""
        reports = self._reports
        
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        return reports[-limit:]

