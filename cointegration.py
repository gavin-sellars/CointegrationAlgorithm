# I think there's an error with the dates, look into that

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.api import OLS
from sklearn.cluster import KMeans
from itertools import combinations
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import scipy.stats as stats
import time
import random
from bayes_opt import BayesianOptimization

warnings.filterwarnings('ignore')

class CointegrationPairsTrader:
    def __init__(self, start_date='2019-01-01', end_date=None, initial_capital=100000, data_file='price_data.pkl'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.initial_capital = initial_capital
        self.data_file = data_file
        self.price_data = None
        self.data = {}
        self.pairs = []
        self.trades = []
        self.all_trades = []  # Initialize all_trades
        self.pair_results = {}  # Initialize pair_results
        self.portfolio_value = []
        self.cash = initial_capital      # available cash
        self.used_capital = 0.0          # capital committed in open trades
        self.equity_curve = []           # list of (date, cash + open P&L)
        self.open_positions = []         # to track which trades are live
        self.notional = 10000.0
        
        # Strategy parameters
        self.lookback_window = 252  # 1 year for cointegration testing
        self.entry_threshold = 2.0  # Z-score threshold for entry
        self.exit_threshold = 0.5   # Z-score threshold for exit
        self.stop_loss = 3.5        # Stop loss threshold
        self.max_positions = 20      # Maximum concurrent positions
        
    def get_expanded_stock_universe(self):
        stocks = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
                'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'CSCO', 'AVGO', 'TXN', 'AMAT', 'LRCX',
                'KLAC', 'MRVL', 'ADI', 'MCHP', 'CTSH', 'INFY', 'ACN', 'SNOW', 'PLTR',
                'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'TWLO', 'ROKU', 'SQ', 'PYPL', 'SHOP'
            ],
            'Financial': [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
                'AXP', 'BLK', 'SCHW', 'CB', 'MMC', 'AON', 'SPGI', 'ICE', 'CME', 'MCO',
                'V', 'MA', 'BRK-B', 'ALL', 'TRV', 'PGR', 'MET', 'PRU', 'AFL', 'AIG'
            ],
            'Healthcare': [
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'MRK', 'BMY', 'AMGN',
                'GILD', 'MDT', 'CVS', 'CI', 'ANTM', 'HUM', 'CNC', 'MOH',
                'BIIB', 'REGN', 'VRTX', 'ILMN', 'ISRG', 'SYK', 'BSX', 'ZBH', 'EW',
                'BAX', 'BDX', 'VAR', 'DXCM', 'ALGN', 'IDXX', 'IQV', 'MTD', 'DHR', 'A'
            ],
            'Consumer_Discretionary': [
                'HD', 'MCD', 'SBUX', 'LOW', 'TJX', 'BKNG', 'NKE', 'CMG',
                'ORLY', 'AZO', 'ROST', 'YUM', 'GM', 'F', 'APTV', 'LVS', 'MGM', 'WYNN',
                'MAR', 'HLT', 'CCL', 'NCLH', 'RCL', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR'
            ],
            'Consumer_Staples': [
                'WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'K',
                'HSY', 'MKC', 'CPB', 'CAG', 'SJM', 'HRL', 'TSN', 'ADM', 'BG',
                'KR', 'SYY', 'DG', 'DLTR', 'WBA', 'TAP', 'STZ', 'DEO',
                'PM', 'MO', 'BTI', 'KHC'
            ],
            'Industrial': [
                'BA', 'CAT', 'GE', 'MMM', 'HON', 'UNP', 'UPS', 'RTX', 'LMT', 'NOC',
                'GD', 'LHX', 'TXT', 'ETN', 'EMR', 'ITW', 'PH', 'CMI', 'DE', 'DOV',
                'IR', 'ROK', 'FTV', 'XYL', 'AME', 'ROP', 'IEX', 'PNR', 'SWK', 'FAST',
                'PCAR', 'CSX', 'NSC', 'KNX', 'CHRW', 'EXPD', 'JBHT', 'ODFL', 'FDX', 'DAL'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'PXD', 'KMI',
                'OKE', 'WMB', 'EPD', 'ET', 'MPLX', 'PAA', 'BKR', 'HAL', 'DVN', 'FANG',
                'MRO', 'APA', 'OXY', 'HES', 'CTRA', 'EQT', 'COG', 'CNX', 'RRC', 'SM'
            ],
            'Materials': [
                'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'CTVA', 'DD', 'ECL', 'FMC', 'LYB',
                'EMN', 'ALB', 'CE', 'BALL', 'AVY', 'PKG', 'IP', 'WRK', 'NUE', 'STLD',
                'RS', 'CMC', 'MLM', 'VMC', 'X', 'CLF', 'AA', 'CENX'
            ],
            'Utilities': [
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED',
                'EIX', 'WEC', 'PPL', 'FE', 'ES', 'DTE', 'ETR', 'CMS', 'CNP', 'ATO',
                'NI', 'LNT', 'EVRG', 'PNW', 'AES', 'VST', 'AWK', 'WTR'
            ],
            'Real_Estate': [
                'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA', 'EXR',
                'AVB', 'EQR', 'VTR', 'SBAC', 'BXP', 'ARE', 'VNO', 'SLG', 'HPP', 'FRT',
                'REG', 'KIM', 'UDR', 'CPT', 'ELS', 'MAA', 'ESS', 'HST', 'PK'
            ]
        }
        
        # Additional stocks for more diversity
        additional_stocks = [
            # Small/Mid cap tech
            'APPS', 'ZI', 'PLAN', 'SMAR', 'TENB', 'ESTC', 'PCTY', 'NEWR', 'SUMO',
        ]
        
        # Flatten the main dictionary and add additional stocks
        all_stocks = []
        for sector_stocks in stocks.values():
            all_stocks.extend(sector_stocks)
        all_stocks.extend(additional_stocks)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_stocks = []
        for stock in all_stocks:
            if stock not in seen:
                seen.add(stock)
                unique_stocks.append(stock)
        
        print(f"Total stock universe: {len(unique_stocks)} stocks")
        return unique_stocks, stocks
    
    def load_or_download_data(self, stock_list, batch_size=10, delay=1.0, use_existing=True):
        """
        Load existing data if available, or download new data if requested.
        """
        # Try to load existing pickle
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    self.price_data = pickle.load(f)
                print(f"Loaded cached data: {self.price_data.shape[1]} tickers, {len(self.price_data)} rows.")

                if use_existing:
                    return True

            except (EOFError, pickle.UnpicklingError):
                print(f"Warning: '{self.data_file}' is empty or corrupted. Starting fresh.")
                self.price_data = pd.DataFrame()
        else:
            self.price_data = pd.DataFrame()

        print(f"Downloading data from {self.start_date.date()} to {self.end_date.date()}...")

        any_success = False

        # Download in batches if needed
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i : i + batch_size]

            for ticker in batch:
                try:
                    time.sleep(delay + random.uniform(0, 0.5))

                    df = yf.Ticker(ticker).history(
                        start=self.start_date.strftime('%Y-%m-%d'),
                        end=self.end_date.strftime('%Y-%m-%d'),
                        auto_adjust=True,
                        prepost=True
                    )['Close']

                    if not df.empty and len(df) >= 10:
                        df = df.rename(ticker)

                        if self.price_data is not None and not self.price_data.empty:
                            self.price_data = pd.concat([self.price_data, df], axis=1)
                            self.price_data = self.price_data.loc[:, ~self.price_data.columns.duplicated()]
                        else:
                            self.price_data = pd.DataFrame(df)

                        self.price_data = self.price_data.dropna(axis=1, how='all')
                        self.price_data = self.price_data.sort_index()

                        with open(self.data_file, 'wb') as f:
                            pickle.dump(self.price_data, f)

                        print(f"Downloaded & saved: {ticker} ({len(df)} rows). Now have {self.price_data.shape[1]} tickers total.")
                        any_success = True
                    else:
                        print(f"No data (or too few rows) for {ticker}. Skipping.")
                except Exception as e:
                    print(f"Error downloading {ticker}: {e}. Skipping.")

            if i + batch_size < len(stock_list):
                time.sleep(delay)

        if any_success:
            return True
        else:
            print("No valid data downloaded.")
            return False

    def find_cointegrated_pairs(self, sector_dict, max_pairs_to_test=5000):
        """
        Efficiently find cointegrated pairs with sampling to manage computation
        """
        # Filter out columns with insufficient data
        min_data_points = 252  # 1 year of data
        valid_stocks = []
        
        for stock in self.price_data.columns:
            if len(self.price_data[stock].dropna()) >= min_data_points:
                valid_stocks.append(stock)
        
        print(f"\nFiltered to {len(valid_stocks)} stocks with sufficient data (min {min_data_points} points)")
        
        if len(valid_stocks) < 2:
            print("Not enough stocks with sufficient data for pair analysis")
            return []
        
        # Calculate total possible pairs
        total_possible_pairs = len(valid_stocks) * (len(valid_stocks) - 1) // 2
        print(f"Total possible pairs: {total_possible_pairs:,}")
        
        if total_possible_pairs > max_pairs_to_test:
            print(f"Sampling {max_pairs_to_test:,} pairs to test (to manage computation time)")
            # Randomly sample pairs, but bias toward cross-sector pairs
            all_pairs = list(combinations(valid_stocks, 2))
            
            # Separate cross-sector and same-sector pairs
            cross_sector_pairs = []
            same_sector_pairs = []
            
            for stock1, stock2 in all_pairs:
                sector1 = self.get_stock_sector(stock1, sector_dict)
                sector2 = self.get_stock_sector(stock2, sector_dict)
                
                if sector1 != sector2:
                    cross_sector_pairs.append((stock1, stock2))
                else:
                    same_sector_pairs.append((stock1, stock2))
            
            # Sample more cross-sector pairs (70%) and fewer same-sector pairs (30%)
            n_cross = min(int(max_pairs_to_test * 0.7), len(cross_sector_pairs))
            n_same = min(max_pairs_to_test - n_cross, len(same_sector_pairs))
            
            pairs_to_test = (random.sample(cross_sector_pairs, n_cross) + 
                           random.sample(same_sector_pairs, n_same))
            
            print(f"Testing {len(pairs_to_test):,} pairs ({n_cross:,} cross-sector, {n_same:,} same-sector)")
        else:
            pairs_to_test = list(combinations(valid_stocks, 2))
            print(f"Testing all {len(pairs_to_test):,} pairs")
        
        potential_pairs = []
        tested_count = 0
        
        for stock1, stock2 in pairs_to_test:
            tested_count += 1
            
            if tested_count % 500 == 0:
                print(f"Tested {tested_count:,}/{len(pairs_to_test):,} pairs, found {len(potential_pairs)} cointegrated pairs")
            
            try:
                # Get aligned price data
                common_data = self.price_data[[stock1, stock2]].dropna()
                if len(common_data) < 252:  # Need at least 1 year of data
                    continue
                
                p1 = common_data[stock1]
                p2 = common_data[stock2]
                
                # Skip if prices are too similar (likely same stock or error)
                if abs(p1.mean() - p2.mean()) / max(p1.mean(), p2.mean()) < 0.01:
                    continue
                
                # Test for cointegration
                coint_score, p_value, _ = coint(p1, p2)
                
                if p_value < 0.05:  # Significant cointegration
                    # Calculate additional metrics
                    correlation = p1.corr(p2)
                    
                    # Skip pairs with too high correlation (likely same sector/related)
                    if abs(correlation) > 0.95:
                        continue
                    
                    # Calculate spread statistics
                    try:
                        model = OLS(p1, p2).fit()
                        spread = p1 - model.params[0] * p2
                        
                        # Test spread for stationarity
                        adf_stat, adf_p_value, _, _, _, _ = adfuller(spread.dropna())
                        
                        if adf_p_value < 0.05:  # Spread is stationary
                            sector1 = self.get_stock_sector(stock1, sector_dict)
                            sector2 = self.get_stock_sector(stock2, sector_dict)
                            cross_sector = sector1 != sector2
                            
                            potential_pairs.append({
                                'stock1': stock1,
                                'stock2': stock2,
                                'coint_score': coint_score,
                                'p_value': p_value,
                                'correlation': correlation,
                                'spread_mean': spread.mean(),
                                'spread_std': spread.std(),
                                'cross_sector': cross_sector,
                                'sector1': sector1,
                                'sector2': sector2,
                                'hedge_ratio': model.params[0],
                                'adf_p_value': adf_p_value,
                                'r_squared': model.rsquared
                            })
                    except Exception as e:
                        # Skip if OLS fails
                        continue
                
            except Exception as e:
                # Skip problematic pairs
                continue
        
        print(f"\nCointegration testing complete!")
        print(f"Found {len(potential_pairs)} cointegrated pairs from {tested_count:,} tests")
        
        if len(potential_pairs) == 0:
            print("No cointegrated pairs found")
            return []
        
        # Sort and select best pairs
        pairs_df = pd.DataFrame(potential_pairs)
        
        # Create composite score favoring cross-sector pairs and strong cointegration
        pairs_df['score'] = (
            (1 - pairs_df['p_value']) * 2 +  # Strong cointegration
            (1 - pairs_df['adf_p_value']) * 1.5 +  # Stationary spread
            pairs_df['cross_sector'].astype(int) * 1 +  # Cross-sector bonus
            pairs_df['r_squared'] * 0.5  # Model fit
        )
        
        pairs_df = pairs_df.sort_values('score', ascending=False)
        
        # Select top pairs
        top_pairs = pairs_df.head(15)
        
        print(f"\nTOP {len(top_pairs)} COINTEGRATED PAIRS:")
        print("-" * 80)
        for _, pair in top_pairs.iterrows():
            cross_label = "✓ Cross-sector" if pair['cross_sector'] else "Same sector"
            print(f"{pair['stock1']}-{pair['stock2']}: p-val={pair['p_value']:.4f}, "
                  f"corr={pair['correlation']:.3f}, R²={pair['r_squared']:.3f}, {cross_label}")
            if pair['cross_sector']:
                print(f"  Sectors: {pair['sector1']} ↔ {pair['sector2']}")
        
        return top_pairs.to_dict('records')
    
    def get_stock_sector(self, stock, sector_dict):
        """Get the sector for a given stock"""
        for sector, stocks in sector_dict.items():
            if stock in stocks:
                return sector
        return 'Other'
    
    def calculate_z_score(self, prices1, prices2, hedge_ratio, window=20):
        """Calculate rolling z-score of the spread"""
        spread = prices1 - hedge_ratio * prices2
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        z_score = (spread - spread_mean) / spread_std
        return z_score, spread

    def has_sufficient_capital(self, required_capital):
        """Check if we have sufficient capital for a new position"""
        return (self.cash - required_capital) >= 0 and len(self.open_positions) < self.max_positions
    
    def allocate_capital(self, amount):
            """Allocate capital for a new position"""
            if self.has_sufficient_capital(amount):
                self.cash -= amount
                self.used_capital += amount
                return True
            return False

    def free_capital(self, amount):
        """Free up capital when closing a position"""
        self.cash += amount
        self.used_capital -= amount

    def objective_function(self, entry, exit):
        """
        Objective function for Bayesian Optimization.
        It runs a backtest with the given entry/exit thresholds and returns the total P&L.
        """

        # Heavily penalize of entry >= exit
        if exit >= entry:
            return -1e9
        
        # Store original thresholds to restore them later
        original_entry = self.entry_threshold
        original_exit = self.exit_threshold

        # Set new thresholds for this optimization iteration
        self.entry_threshold = entry
        self.exit_threshold = exit

        # Reset capital to ensure a fair comparison
        self.cash = self.initial_capital
        self.used_capital = 0.0
        self.open_positions = []

        all_trades_this_run = []

        # Loop through the pre-selected pairs and run a backtest for each
        for pair_info in self.pairs:
            trades, _, _, _ = self.backtest_pair(pair_info)
            if trades:
                all_trades_this_run.extend(trades)

        # Restore original thresholds
        self.entry_threshold = original_entry
        self.exit_threshold = original_exit
        
        if not all_trades_this_run:
            return 0.0

        total_pnl = sum(t['pnl'] for t in all_trades_this_run)
        
        # Ensure a valid number is returned
        return total_pnl if not np.isnan(total_pnl) else 0.0

    def optimize_parameters(self):
        """
        Use Bayesian Optimization to find the best entry and exit thresholds.
        """
        if not self.pairs:
            print("Cannot run optimization without cointegrated pairs. Find pairs first.")
            return
        print("\n" + "=" * 80)
        print("STARTING BAYESIAN OPTIMIZATION FOR ENTRY/EXIT THRESHOLDS")
        print("=" * 80)
        
        # Define the parameter space (pbounds) for the optimizer
        pbounds = {
            'entry': (1.5, 3.5),  # Explore entry Z-scores between 1.5 and 3.5
            'exit': (0.1, 1.4)    # Explore exit Z-scores between 0.1 and 1.4
        }

        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2  # 2 prints all steps, 1 prints only improvements
        )

        optimizer.maximize(
            init_points=5,  # Number of random exploration steps
            n_iter=15       # Number of optimization steps
        )

        # Get the best parameters found
        best_params = optimizer.max['params']
        best_entry = best_params['entry']
        best_exit = best_params['exit']

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print(f"Optimal Entry Threshold: {best_entry:.4f}")
        print(f"Optimal Exit Threshold: {best_exit:.4f}")
        print(f"Best P&L found during optimization: ${optimizer.max['target']:,.2f}")
        print("=" * 80)

        # Update the class attributes with the new optimal values
        self.entry_threshold = best_entry
        self.exit_threshold = best_exit

    def backtest_pair(self, pair_info, lookback_window=20):
        """
        Backtest a single cointegrated pair with share-based sizing, capital tracking,
        warm-up period, spread threshold, and equity curve construction
        """
        stock1 = pair_info['stock1']
        stock2 = pair_info['stock2']
        hedge_ratio = pair_info['hedge_ratio']

        # Align prices and drop NaNs
        data = self.price_data[[stock1, stock2]].dropna()

        # Want at least a year of data
        if len(data) < 252:
            return [], pd.Series(dtype=float), pd.Series(dtype=float), []
            
        p1 = data[stock1]
        p2 = data[stock2]

        # Compute spread and rolling z-score
        spread = p1 - hedge_ratio * p2
        spread_mean = spread.rolling(window=lookback_window).mean()
        spread_std = spread.rolling(window=lookback_window).std()
        z_score = (spread - spread_mean) / spread_std

        trades = []
        positions = []
        min_periods = 3 * lookback_window
        
        # Track positions for this specific pair
        current_position = 0
        position_entry = None
        position_id = None

        for i in range(len(z_score)):
            date = z_score.index[i]
            current_z = z_score.iloc[i]
            current_spread = spread.iloc[i]
            price1 = p1.iloc[i]
            price2 = p2.iloc[i]

            # Warm-up
            if i < min_periods or pd.isna(current_z):
                positions.append(0)
                continue

            # EXIT logic
            if current_position != 0 and position_entry is not None:
                exit_signal = False
                reason = ''
                
                # Mean reversion
                if abs(current_z) < self.exit_threshold:
                    exit_signal = True
                    reason = 'mean_reversion'
                # Stop loss
                elif (current_position == 1 and current_z > self.stop_loss) or \
                     (current_position == -1 and current_z < -self.stop_loss):
                    exit_signal = True
                    reason = 'stop_loss'
                # Reversal
                elif (current_position == 1 and current_z > self.entry_threshold) or \
                     (current_position == -1 and current_z < -self.entry_threshold):
                    exit_signal = True
                    reason = 'reversal'

                if exit_signal:
                    # Calculate P&L with hedge ratio
                    stock1_pnl = (price1 - position_entry['price1']) * position_entry['shares1']
                    stock2_pnl = (price2 - position_entry['price2']) * position_entry['shares2']
                    
                    if position_entry['position'] == 1:  # Long stock1, short stock2
                        pnl = stock1_pnl - stock2_pnl
                    else:  # Short stock1, long stock2
                        pnl = -stock1_pnl + stock2_pnl

                    trades.append({
                        'pair': f"{stock1}-{stock2}",
                        'entry_date': position_entry['date'],
                        'exit_date': date,
                        'entry_z': position_entry['z'],
                        'exit_z': current_z,
                        'position': position_entry['position'],
                        'pnl': pnl,
                        'notional': position_entry['notional'],
                        'pnl_pct': pnl / position_entry['notional'],
                        'days_held': (date - position_entry['date']).days,
                        'exit_reason': reason
                    })
                    
                    # Free up capital and remove from open positions
                    self.free_capital(position_entry['notional'])
                    if position_id in self.open_positions:
                        self.open_positions.remove(position_id)
                    
                    current_position = 0
                    position_entry = None
                    position_id = None

            # ENTRY logic 
            if current_position == 0:
                if abs(current_z) > self.entry_threshold and abs(current_spread) >= 0.5:
                    required_capital = self.notional
                    
                    # Check if we have sufficient capital and haven't exceeded max positions
                    if self.has_sufficient_capital(required_capital):
                        # Allocate capital
                        if self.allocate_capital(required_capital):
                            # Fixed position sizing with proper hedge ratio application
                            total_exposure = price1 + abs(hedge_ratio) * price2
                            shares1 = self.notional / total_exposure
                            shares2 = abs(hedge_ratio) * shares1
                            
                            pos = -1 if current_z > self.entry_threshold else 1
                            
                            position_entry = {
                                'date': date,
                                'stock1': stock1,
                                'stock2': stock2,
                                'price1': price1,
                                'price2': price2,
                                'shares1': shares1,
                                'shares2': shares2,
                                'position': pos,
                                'notional': self.notional,
                                'z': current_z
                            }
                            current_position = pos
                            
                            # Track this position globally
                            position_id = f"{stock1}-{stock2}-{date}"
                            self.open_positions.append(position_id)

            positions.append(current_position)

        return trades, z_score, spread, positions


    def calculate_performance(self):
        """Generate performance metrics based on trades."""
        if not self.all_trades:
            print("No trades to analyze.")
            return

        trades_df = pd.DataFrame(self.all_trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = (abs(avg_win * winning_trades) / abs(avg_loss * losing_trades)) \
                       if avg_loss < 0 and losing_trades > 0 else float('inf')
        
        # Calculate returns
        total_return = total_pnl / self.initial_capital
        
        # Simple annualized return calculation
        if len(trades_df) > 0:
            start_date = trades_df['entry_date'].min()
            end_date = trades_df['exit_date'].max()
            days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        else:
            annualized_return = 0
        
        # Print results
        print("------------------------------------------------------------")
        print("BACKTEST RESULTS SUMMARY")
        print("------------------------------------------------------------")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Cash: ${self.cash:,.2f}")
        print(f"Used Capital: ${self.used_capital:,.2f}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Annualized Return: {annualized_return*100:.2f}%")
        print("\nTrades Summary:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Concurrent Positions Used: {len(self.open_positions)}")

    def is_cross_sector_trade(self, pair_name):
        """Check if a trade is cross-sector"""
        for pair_info in self.pairs:
            if f"{pair_info['stock1']}-{pair_info['stock2']}" == pair_name:
                return pair_info['cross_sector']
        return False
    
    def calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        if trades_df.empty:
            return 0
        cumulative_pnl = trades_df['pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - rolling_max) / self.initial_capital
        return drawdown.min()

    def run_backtest(self):
        """Run the complete backtest process with proper capital management"""
        print("=" * 80)
        print("STARTING COMPREHENSIVE COINTEGRATION PAIRS TRADING BACKTEST")
        print("=" * 80)
        
        # Reset capital tracking for fresh backtest
        self.cash = self.initial_capital
        self.used_capital = 0.0
        self.open_positions = []
        
        # Step 1: Get stock universe
        stock_list, sector_dict = self.get_expanded_stock_universe()
        
        # Step 2: Load/download data
        print(f"\nLoading price data from {self.start_date.date()} to {self.end_date.date()}...")
        success = self.load_or_download_data(stock_list)
        if not success:
            print("Failed to load sufficient data for backtesting")
            return
        
        print(f"Loaded data for {self.price_data.shape[1]} stocks, {self.price_data.shape[0]} trading days")
        
        # Step 3: Find cointegrated pairs
        print(f"\nSearching for cointegrated pairs...")
        pairs = self.find_cointegrated_pairs(sector_dict)
        if not pairs:
            print("No cointegrated pairs found")
            return
        
        self.pairs = pairs
        print(f"Found {len(pairs)} cointegrated pairs for trading")

        # Step 4: Optimize strategy parameters using Bayesian Optimization
        self.optimize_parameters()
        
        # Step 5: Backtest each pair with the now-optimized parameters

        print(f"\nRunning FINAL backtest with optimized parameters...")

        self.all_trades = []
        self.pair_results = {}
        
        for i, pair in enumerate(self.pairs):
            pair_name = f"{pair['stock1']}-{pair['stock2']}"
            print(f"  Testing pair {i+1}/{len(pairs)}: {pair_name}")
            
            try:
                trades, z_score, spread, positions = self.backtest_pair(pair)
                
                self.pair_results[pair_name] = {
                    'trades': trades,
                    'z_score': z_score,
                    'spread': spread,
                    'positions': positions,
                    'pair_info': pair
                }
                
                # Add trades to master list
                self.all_trades.extend(trades)
                
                if trades:
                    total_pnl = sum([t['pnl'] for t in trades])
                    print(f"    -> {len(trades)} trades, ${total_pnl:.0f} total P&L")
                else:
                    print(f"    -> No trades generated")
                    
            except Exception as e:
                print(f"    -> Error backtesting {pair_name}: {e}")
                self.pair_results[pair_name] = {
                    'trades': [],
                    'z_score': pd.Series(dtype=float),
                    'spread': pd.Series(dtype=float),
                    'positions': [],
                    'pair_info': pair
                }
        
        # Step 5: Calculate and display results
        print(f"\nCalculating performance metrics...")
        self.calculate_performance()
        
        if self.all_trades:
            print(f"\nBacktest completed successfully!")
            print(f"Total pairs tested: {len(pairs)}")
            print(f"Total trades executed: {len(self.all_trades)}")
            print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        else:
            print(f"\nBacktest completed but no trades were executed")
            print("Consider adjusting strategy parameters (thresholds, lookback window, etc.)")
    
    def plot_results(self):
        """Create comprehensive visualizations"""
        if not self.all_trades:
            print("No trades to visualize!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cointegration Pairs Trading Strategy - Comprehensive Results', fontsize=16)
        
        trades_df = pd.DataFrame(self.all_trades)
        
        # 1. Cumulative P&L
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['cumulative_return'] = trades_df['cumulative_pnl'] / self.initial_capital
        
        axes[0, 0].plot(trades_df.index, trades_df['cumulative_return'] * 100, linewidth=2, color='darkblue')
        axes[0, 0].set_title('Cumulative Returns (%)')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. P&L Distribution
        axes[0, 1].hist(trades_df['pnl'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[0, 1].set_title('P&L Distribution')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Win Rate by Pair
        pair_stats = {}
        for pair_name, result in self.pair_results.items():
            if result['trades'] and len(result['trades']) >= 3:
                wins = len([t for t in result['trades'] if t['pnl'] > 0])
                total = len(result['trades'])
                pair_stats[pair_name] = wins / total * 100
        
        if pair_stats:
            # Sort and take top 10
            sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            pairs = [p[0] for p in sorted_pairs]
            win_rates = [p[1] for p in sorted_pairs]
            
            colors = ['lightgreen' if self.is_cross_sector_trade(p) else 'lightcoral' for p in pairs]
            axes[0, 2].barh(pairs, win_rates, color=colors, alpha=0.7)
            axes[0, 2].set_title('Win Rate by Pair (%) - Green=Cross-Sector')
            axes[0, 2].set_xlabel('Win Rate (%)')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'No pairs with 3+ trades', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Win Rate by Pair')
        
        # 4. Holding Period Analysis
        axes[1, 0].hist(trades_df['days_held'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Trade Holding Period Distribution')
        axes[1, 0].set_xlabel('Days Held')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Monthly Returns Analysis
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['month'] = trades_df['entry_date'].dt.to_period('M')
        monthly_returns = trades_df.groupby('month')['pnl'].sum()
        
        if len(monthly_returns) > 12:
            # Create heatmap for longer periods
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            
            heatmap_data = np.zeros((len(years), 12))
            for i, year in enumerate(years):
                for j, month in enumerate(months):
                    try:
                        period = pd.Period(f"{year}-{month:02d}")
                        if period in monthly_returns.index:
                            heatmap_data[i, j] = monthly_returns[period]
                    except:
                        heatmap_data[i, j] = 0
            
            im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
            axes[1, 1].set_title('Monthly P&L Heatmap')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Year')
            axes[1, 1].set_xticks(range(12))
            axes[1, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            axes[1, 1].set_yticks(range(len(years)))
            axes[1, 1].set_yticklabels(years)
            plt.colorbar(im, ax=axes[1, 1])
        else:
            # Simple bar chart for shorter periods
            x_pos = range(len(monthly_returns))
            axes[1, 1].bar(x_pos, monthly_returns.values, alpha=0.7, color='steelblue')
            axes[1, 1].set_title('Monthly P&L')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('P&L ($)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([str(m) for m in monthly_returns.index], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Z-Score example (best performing pair)
        if self.pair_results:
            best_pair = None
            best_pnl = float('-inf')
            
            for pair_name, result in self.pair_results.items():
                if result['trades']:
                    total_pnl = sum([t['pnl'] for t in result['trades']])
                    if total_pnl > best_pnl:
                        best_pnl = total_pnl
                        best_pair = (pair_name, result)
            
            if best_pair:
                pair_name, pair_data = best_pair
                z_score = pair_data['z_score'].dropna()
                
                if len(z_score) > 0:
                    # Sample the data if too many points
                    if len(z_score) > 1000:
                        sample_indices = np.linspace(0, len(z_score)-1, 1000, dtype=int)
                        z_score = z_score.iloc[sample_indices]
                    
                    axes[1, 2].plot(z_score.index, z_score.values, alpha=0.7, color='blue', linewidth=1)
                    axes[1, 2].axhline(y=self.entry_threshold, color='red', linestyle='--', alpha=0.7, label='Entry Threshold')
                    axes[1, 2].axhline(y=-self.entry_threshold, color='red', linestyle='--', alpha=0.7)
                    axes[1, 2].axhline(y=self.exit_threshold, color='green', linestyle='--', alpha=0.7, label='Exit Threshold')
                    axes[1, 2].axhline(y=-self.exit_threshold, color='green', linestyle='--', alpha=0.7)
                    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    axes[1, 2].set_title(f'Z-Score: {pair_name} (Best Performer)')
                    axes[1, 2].set_xlabel('Date')
                    axes[1, 2].set_ylabel('Z-Score')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
                    
                    # Rotate x-axis labels for better readability
                    for tick in axes[1, 2].get_xticklabels():
                        tick.set_rotation(45)
                else:
                    axes[1, 2].text(0.5, 0.5, 'No Z-Score data available', 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=axes[1, 2].transAxes)
                    axes[1, 2].set_title('Z-Score Analysis')
            else:
                axes[1, 2].text(0.5, 0.5, 'No profitable pairs found', 
                              horizontalalignment='center', verticalalignment='center',
                              transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Z-Score Analysis')
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_trades(self, n=10):
        """Print top n and worst trades with dollar and percent P&L."""
        if not self.all_trades:
            print("No trades executed.")
            return

        # Ensure pnl_pct exists on each trade
        for tr in self.all_trades:
            if 'pnl_pct' not in tr and 'notional' in tr and tr['notional'] != 0:
                tr['pnl_pct'] = tr['pnl'] / tr['notional']
            elif 'pnl_pct' not in tr:
                tr['pnl_pct'] = 0

        trades_df = pd.DataFrame(self.all_trades)
        
        # Top trades
        if len(trades_df) > 0:
            top = trades_df.nlargest(min(n, len(trades_df)), 'pnl')
            print(f"\nTop {len(top)} Trades:")
            print("-" * 120)
            for _, tr in top.iterrows():
                pct = tr.get('pnl_pct', 0)
                entry_date = tr['entry_date'].date() if hasattr(tr['entry_date'], 'date') else str(tr['entry_date'])[:10]
                exit_date = tr['exit_date'].date() if hasattr(tr['exit_date'], 'date') else str(tr['exit_date'])[:10]
                print(f"{tr['pair']:<12} | Entry: {entry_date} | Exit: {exit_date} | "
                      f"P&L: ${tr['pnl']:>8.2f} ({pct:>6.2%}) | Held: {tr['days_held']:>3}d | "
                      f"Reason: {tr.get('exit_reason', 'N/A')}")

            # Worst trades
            worst = trades_df.nsmallest(min(5, len(trades_df)), 'pnl')
            print(f"\nWorst {len(worst)} Trades:")
            print("-" * 120)
            for _, tr in worst.iterrows():
                pct = tr.get('pnl_pct', 0)
                entry_date = tr['entry_date'].date() if hasattr(tr['entry_date'], 'date') else str(tr['entry_date'])[:10]
                exit_date = tr['exit_date'].date() if hasattr(tr['exit_date'], 'date') else str(tr['exit_date'])[:10]
                print(f"{tr['pair']:<12} | Entry: {entry_date} | Exit: {exit_date} | "
                      f"P&L: ${tr['pnl']:>8.2f} ({pct:>6.2%}) | Held: {tr['days_held']:>3}d | "
                      f"Reason: {tr.get('exit_reason', 'N/A')}")
    
    def print_strategy_analysis(self):
        """Print detailed strategy analysis"""
        if not self.all_trades:
            print("No trades to analyze.")
            return
        
        print("\n" + "=" * 80)
        print("DETAILED STRATEGY ANALYSIS")
        print("=" * 80)
        
        trades_df = pd.DataFrame(self.all_trades)
        
        # Cross-sector vs Same-sector performance
        cross_sector_trades = [t for t in self.all_trades if self.is_cross_sector_trade(t['pair'])]
        same_sector_trades = [t for t in self.all_trades if not self.is_cross_sector_trade(t['pair'])]
        
        if cross_sector_trades and same_sector_trades:
            cross_pnl = sum([t['pnl'] for t in cross_sector_trades])
            same_pnl = sum([t['pnl'] for t in same_sector_trades])
            cross_win_rate = len([t for t in cross_sector_trades if t['pnl'] > 0]) / len(cross_sector_trades)
            same_win_rate = len([t for t in same_sector_trades if t['pnl'] > 0]) / len(same_sector_trades)
            
            print("CROSS-SECTOR vs SAME-SECTOR ANALYSIS:")
            print(f"Cross-Sector: {len(cross_sector_trades)} trades, ${cross_pnl:.2f} P&L, {cross_win_rate:.1%} win rate")
            print(f"Same-Sector:  {len(same_sector_trades)} trades, ${same_pnl:.2f} P&L, {same_win_rate:.1%} win rate")
            print(f"Cross-sector advantage: ${cross_pnl - same_pnl:.2f}")
        elif cross_sector_trades:
            cross_pnl = sum([t['pnl'] for t in cross_sector_trades])
            cross_win_rate = len([t for t in cross_sector_trades if t['pnl'] > 0]) / len(cross_sector_trades)
            print("CROSS-SECTOR ANALYSIS:")
            print(f"Cross-Sector: {len(cross_sector_trades)} trades, ${cross_pnl:.2f} P&L, {cross_win_rate:.1%} win rate")
            print("No same-sector trades for comparison")
        elif same_sector_trades:
            same_pnl = sum([t['pnl'] for t in same_sector_trades])
            same_win_rate = len([t for t in same_sector_trades if t['pnl'] > 0]) / len(same_sector_trades)
            print("SAME-SECTOR ANALYSIS:")
            print(f"Same-Sector: {len(same_sector_trades)} trades, ${same_pnl:.2f} P&L, {same_win_rate:.1%} win rate")
            print("No cross-sector trades for comparison")
        
        # Exit reason performance
        if 'exit_reason' in trades_df.columns:
            print(f"\nEXIT REASON PERFORMANCE:")
            exit_analysis = trades_df.groupby('exit_reason').agg({
                'pnl': ['count', 'sum', 'mean'],
                'days_held': 'mean'
            }).round(2)
            
            # Flatten column names
            exit_analysis.columns = ['Count', 'Total_PnL', 'Avg_PnL', 'Avg_Days_Held']
            print(exit_analysis)
        
        # Best performing stock pairs
        print(f"\nSTOCK PAIR PERFORMANCE (min 3 trades):")
        pair_performance = {}
        for pair_name, result in self.pair_results.items():
            if result['trades'] and len(result['trades']) >= 3:
                trades = result['trades']
                total_pnl = sum([t['pnl'] for t in trades])
                win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
                avg_pnl = total_pnl / len(trades)
                avg_days = np.mean([t['days_held'] for t in trades])
                
                pair_performance[pair_name] = {
                    'trades': len(trades),
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'avg_days': avg_days,
                    'cross_sector': self.is_cross_sector_trade(pair_name)
                }
        
        if pair_performance:
            # Sort by total P&L
            sorted_pairs = sorted(pair_performance.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
            
            print("Top performing pairs:")
            print("-" * 100)
            for pair_name, stats in sorted_pairs[:10]:
                cross_flag = "🔀" if stats['cross_sector'] else "📊"
                print(f"{cross_flag} {pair_name:<12}: {stats['trades']:>2} trades, ${stats['total_pnl']:>7.0f} total, "
                      f"${stats['avg_pnl']:>6.0f} avg, {stats['win_rate']:>5.1%} win, {stats['avg_days']:>4.0f} days")
        else:
            print("No pairs with 3+ trades found.")

    
            
    def monte_carlo_simulation(self, n_simulations=1000, random_seed=42):
        """
        Run a Monte Carlo simulation by resampling trade P&L with replacement
        to assess strategy robustness. Produces summary statistics and plots.
        """
        if not self.all_trades:
            print("No trades available for Monte Carlo simulation.")
            return

        np.random.seed(random_seed)
        trades_df = pd.DataFrame(self.all_trades)

        # Extract actual P&L series
        pnl_series = trades_df['pnl'].values
        n_trades = len(pnl_series)

        final_returns = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Resample trade P&Ls with replacement
            simulated_pnls = np.random.choice(pnl_series, size=n_trades, replace=True)
            cumulative = simulated_pnls.cumsum()

            # Final P&L
            final_returns.append(cumulative[-1] / self.initial_capital)

            # Max drawdown
            rolling_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - rolling_max)
            max_drawdowns.append(drawdown.min() / self.initial_capital)

            # Sharpe ratio (using per-trade returns, not time annualized)
            mean_return = np.mean(simulated_pnls)
            std_return = np.std(simulated_pnls)
            sharpe = mean_return / std_return if std_return > 0 else 0
            sharpe_ratios.append(sharpe)

        # Convert to numpy arrays
        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        # Print summary
        print("\n" + "=" * 80)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 80)
        print(f"Simulations run: {n_simulations}")
        print(f"Mean Final Return: {np.mean(final_returns)*100:.2f}%")
        print(f"Median Final Return: {np.median(final_returns)*100:.2f}%")
        print(f"5th–95th Percentile Range: {np.percentile(final_returns, 5)*100:.2f}% → {np.percentile(final_returns, 95)*100:.2f}%")
        print(f"Mean Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
        print(f"Mean Max Drawdown: {np.mean(max_drawdowns)*100:.2f}%")
        print("=" * 80)

        # Plot histogram of outcomes
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Monte Carlo Simulation - Strategy Robustness', fontsize=16)

        # Final return distribution
        axes[0].hist(final_returns*100, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(final_returns*100), color='red', linestyle='--', label='Mean')
        axes[0].set_title('Final Return Distribution (%)')
        axes[0].set_xlabel('Final Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()

        # Sharpe ratio distribution
        axes[1].hist(sharpe_ratios, bins=40, color='seagreen', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(sharpe_ratios), color='red', linestyle='--', label='Mean')
        axes[1].set_title('Sharpe Ratio Distribution')
        axes[1].set_xlabel('Sharpe Ratio')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()

        # Max drawdown distribution
        axes[2].hist(max_drawdowns*100, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(max_drawdowns*100), color='red', linestyle='--', label='Mean')
        axes[2].set_title('Max Drawdown Distribution (%)')
        axes[2].set_xlabel('Max Drawdown (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()

        plt.tight_layout()
        plt.show()


# Run the backtest
if __name__ == "__main__":
    print("LAUNCHING COINTEGRATION STRATEGY")
    
    # Initialize the strategy
    trader = CointegrationPairsTrader(
        start_date='2015-01-01',
        end_date='2025-06-17',
        initial_capital=100000
    )
    
    # Run comprehensive backtest
    trader.run_backtest()
    
    # Show detailed analysis
    if hasattr(trader, 'all_trades') and trader.all_trades:
        trader.print_detailed_trades(8)
        trader.print_strategy_analysis()
        trader.monte_carlo_simulation(n_simulations=5000)
        trader.plot_results()

        print("Thank you!")