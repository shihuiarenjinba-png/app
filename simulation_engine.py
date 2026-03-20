import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.decomposition import PCA
from datetime import datetime

from factor_data_loader import load_factor_dataset

# 🔻追加: 多言語辞書モジュールの読み込み
try:
    from i18n import ja, en
except ImportError:
    pass # 実行環境によってはapp.py側でエラーハンドリングするためここはpass

# 翻訳呼び出し用のヘルパー関数 (エンジン内用)
def get_text(key, lang='JA'):
    lang_upper = str(lang).upper()
    if lang_upper == 'JA':
        return ja.TEXTS.get(key, key)
    else:
        return en.TEXTS.get(key, key)

# =========================================================
# 🛠️ Class Definitions (Brain: V20 - Traditional GBM & No Magic Numbers)
# =========================================================

class MarketDataEngine:
    """Manages market data, factors, and benchmarks."""
    def __init__(self):
        self.start_date = "2000-01-01"
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.usdjpy_cache = None
        self.us_sector_proxies = {
            'technology': 'XLK',
            'communication services': 'XLC',
            'consumer cyclical': 'XLY',
            'consumer defensive': 'XLP',
            'financial services': 'XLF',
            'healthcare': 'XLV',
            'industrials': 'XLI',
            'energy': 'XLE',
            'basic materials': 'XLB',
            'utilities': 'XLU',
            'real estate': 'XLRE',
        }
        self.industry_keyword_proxies = {
            'semiconductor': 'SOXX',
            'chip': 'SOXX',
            'software': 'IGV',
            'internet': 'XLC',
            'biotech': 'XBI',
            'bank': 'KBE',
            'insurance': 'KIE',
        }

    def validate_tickers(self, input_dict):
        """Check if tickers exist."""
        valid_data = {}
        invalid_tickers = []
        status_text = st.empty()
        
        for ticker, weight in input_dict.items():
            try:
                # 💡上場廃止銘柄なども検知できるよう、取得期間を "max" に変更して存在確認
                test_data = yf.download(ticker, period="max", progress=False)
                if not test_data.empty:
                    valid_data[ticker] = {'name': ticker, 'weight': weight}
                    status_text.text(f"✅ OK: {ticker}")
                else:
                    invalid_tickers.append(ticker)
            except Exception:
                invalid_tickers.append(ticker)
        
        status_text.empty()
        return valid_data, invalid_tickers

    def _get_usdjpy(self):
        """Fetch JPY rate with cache."""
        if self.usdjpy_cache is not None:
            return self.usdjpy_cache
        try:
            raw = yf.download("JPY=X", start=self.start_date, end=self.end_date, interval="1mo", auto_adjust=True, progress=False)
            
            if isinstance(raw, pd.DataFrame):
                if 'Close' in raw.columns:
                    usdjpy = raw['Close']
                else:
                    usdjpy = raw.iloc[:, 0]
            else:
                usdjpy = raw

            if isinstance(usdjpy, pd.DataFrame):
                usdjpy = usdjpy.iloc[:, 0]

            # Pandas最新版対応: 'M' -> 'ME'
            usdjpy = usdjpy.resample('ME').last().ffill()
            if usdjpy.index.tz is not None: 
                usdjpy.index = usdjpy.index.tz_localize(None)
            
            self.usdjpy_cache = usdjpy
            return usdjpy
        except Exception:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=3600*24*7)
    def fetch_french_factors(_self, region='US'):
        """Fetch Fama-French 5 Factors."""
        try:
            return load_factor_dataset(
                region=region,
                start_date=_self.start_date,
                end_date=_self.end_date,
            )
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600*24)
    def fetch_historical_prices(_self, tickers, base_currency='JPY'):
        """Fetch stock prices. UPDATED: Added multi-currency support and Union alignment."""
        try:
            raw_data = yf.download(tickers, start=_self.start_date, end=_self.end_date, interval="1mo", auto_adjust=True, progress=False)
            data = pd.DataFrame()

            if len(tickers) == 1:
                ticker = tickers[0]
                if isinstance(raw_data, pd.Series):
                    data[ticker] = raw_data
                elif isinstance(raw_data, pd.DataFrame):
                    # 株式分割バグを防ぐため、生のCloseではなく Adj Close（調整後終値）を最優先で取得
                    if 'Adj Close' in raw_data.columns:
                        data[ticker] = raw_data['Adj Close']
                    elif 'Close' in raw_data.columns:
                        data[ticker] = raw_data['Close']
                    else:
                        data[ticker] = raw_data.iloc[:, 0]
            else:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    try:
                        data = raw_data.xs('Adj Close', axis=1, level=0, drop_level=True)
                    except KeyError:
                        try:
                            data = raw_data.xs('Close', axis=1, level=0, drop_level=True)
                        except:
                            data = raw_data.iloc[:, :len(tickers)]
                            data.columns = tickers
                else:
                    data = raw_data

            data = data.resample('ME').last().ffill()
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            usdjpy = _self._get_usdjpy()
            if not usdjpy.empty:
                usdjpy = usdjpy.reindex(data.index, method='ffill')
                data_converted = data.copy()
                
                for col in data.columns:
                    is_japan = str(col).endswith(".T") or str(col) in ["^N225", "^TPX", "1306.T"]
                    
                    if base_currency == 'JPY':
                        if not is_japan:
                            data_converted[col] = data[col] * usdjpy
                            
                    elif base_currency == 'USD':
                        if is_japan:
                            data_converted[col] = data[col] / usdjpy
                            
                data_final = data_converted
            else:
                data_final = data

            returns = data_final.pct_change().dropna(how='all')
            
            valid_cols = [c for c in returns.columns if c in tickers]
            if valid_cols:
                returns = returns[valid_cols]
            
            return returns
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600*24)
    def fetch_benchmark_data(_self, ticker, is_jpy_asset=False, base_currency='JPY'):
        """Fetch benchmark. UPDATED: Added multi-currency support."""
        try:
            raw_data = yf.download(ticker, start=_self.start_date, end=_self.end_date, interval="1mo", auto_adjust=True, progress=False)
            data = pd.Series(dtype=float)
            if isinstance(raw_data, pd.DataFrame):
                if 'Adj Close' in raw_data.columns:
                    data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns:
                    data = raw_data['Close']
                elif isinstance(raw_data.columns, pd.MultiIndex):
                     try: data = raw_data.xs('Adj Close', axis=1, level=0, drop_level=True)
                     except:
                         try: data = raw_data.xs('Close', axis=1, level=0, drop_level=True)
                         except: data = raw_data.iloc[:, 0]
                else:
                    data = raw_data.iloc[:, 0]
            else:
                data = raw_data

            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]

            data = data.resample('ME').last().ffill()
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            usdjpy = _self._get_usdjpy()
            if not usdjpy.empty:
                usdjpy = usdjpy.reindex(data.index, method='ffill')
                
                if base_currency == 'JPY':
                    if not is_jpy_asset:
                        data = data * usdjpy
                elif base_currency == 'USD':
                    if is_jpy_asset:
                        data = data / usdjpy
            
            return data.pct_change().dropna()
        except:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=3600*24*7)
    def get_ticker_profile(_self, ticker):
        profile = {
            'ticker': ticker,
            'quoteType': None,
            'sector': None,
            'industry': None,
            'longName': None,
        }
        try:
            info = yf.Ticker(ticker).info or {}
            profile['quoteType'] = info.get('quoteType')
            profile['sector'] = info.get('sector')
            profile['industry'] = info.get('industry')
            profile['longName'] = info.get('longName') or info.get('shortName')
        except Exception:
            return profile
        return profile

    def suggest_proxy_ticker(self, ticker, region='US', benchmark_ticker='SPY'):
        upper_ticker = str(ticker).upper()
        if upper_ticker.endswith('.T'):
            return '1306.T', 'Japan broad-market proxy'

        profile = self.get_ticker_profile(ticker)
        quote_type = str(profile.get('quoteType') or '').lower()
        sector = str(profile.get('sector') or '').lower()
        industry = str(profile.get('industry') or '').lower()
        long_name = str(profile.get('longName') or '').lower()

        if 'etf' in quote_type:
            if region == 'Japan':
                return '1306.T', 'Japan ETF proxy'
            if region == 'Global':
                return 'VT', 'Global ETF proxy'
            return benchmark_ticker, 'US ETF proxy'

        for keyword, proxy in self.industry_keyword_proxies.items():
            if keyword in industry or keyword in long_name:
                return proxy, f'Industry keyword match: {keyword}'

        if sector in self.us_sector_proxies:
            return self.us_sector_proxies[sector], f'Sector match: {sector}'

        if region == 'Japan':
            return '1306.T', 'Japan broad-market proxy'
        if region == 'Global':
            return 'VT', 'Global broad-market proxy'
        return benchmark_ticker, 'US broad-market proxy'

    def build_proxy_extended_history(self, returns_df, tickers, base_currency='JPY', region='US', benchmark_ticker='SPY'):
        if returns_df.empty:
            return returns_df, {}

        proxy_assignments = {}
        proxy_universe = []
        for ticker in tickers:
            actual = returns_df[ticker] if ticker in returns_df.columns else pd.Series(dtype=float)
            missing_months = int(actual.isna().sum()) if not actual.empty else 0
            if actual.empty or missing_months > 0:
                proxy_ticker, reason = self.suggest_proxy_ticker(ticker, region=region, benchmark_ticker=benchmark_ticker)
                if proxy_ticker and proxy_ticker != ticker:
                    proxy_assignments[ticker] = {'proxy_ticker': proxy_ticker, 'reason': reason}
                    proxy_universe.append(proxy_ticker)

        if not proxy_assignments:
            return returns_df, {}

        proxy_universe = sorted(set(proxy_universe))
        proxy_returns = self.fetch_historical_prices(proxy_universe, base_currency=base_currency)
        extended = returns_df.copy()
        metadata = {}

        for ticker, assignment in proxy_assignments.items():
            proxy_ticker = assignment['proxy_ticker']
            if proxy_ticker not in proxy_returns.columns or ticker not in extended.columns:
                continue

            actual = extended[ticker]
            proxy_series = proxy_returns[proxy_ticker].reindex(extended.index)
            missing_mask = actual.isna()
            proxy_fill_mask = missing_mask & proxy_series.notna()
            if proxy_fill_mask.sum() == 0:
                continue

            extended.loc[proxy_fill_mask, ticker] = proxy_series.loc[proxy_fill_mask]
            metadata[ticker] = {
                'proxy_ticker': proxy_ticker,
                'reason': assignment['reason'],
                'proxy_months_used': int(proxy_fill_mask.sum()),
                'first_actual_date': str(actual.first_valid_index().date()) if actual.notna().any() else None,
            }

        return extended, metadata

class PortfolioAnalyzer:
    @staticmethod
    def _should_rebalance(date, rebalance_freq):
        if rebalance_freq == 'M':
            return True
        if rebalance_freq == 'Q' and date.month in [3, 6, 9, 12]:
            return True
        if rebalance_freq == 'Y' and date.month == 12:
            return True
        return False

    @staticmethod
    def _normalized_available_weights(norm_weights, available_assets):
        available_weights = pd.Series({asset: norm_weights[asset] for asset in available_assets}, dtype=float)
        if available_weights.sum() <= 0:
            return pd.Series(dtype=float)
        return available_weights / available_weights.sum()

    @staticmethod
    def create_synthetic_history(returns_df, weights_dict, benchmark_ret=None, rebalance_freq='M', analysis_mode='strict'):
        valid_tickers = [t for t in weights_dict.keys() if t in returns_df.columns]
        if not valid_tickers:
            return pd.Series(dtype=float), {}, {}

        filtered_weights = {k: weights_dict[k] for k in valid_tickers}
        total_weight = sum(filtered_weights.values())
        norm_weights = {k: v / total_weight for k, v in filtered_weights.items()}

        working_returns = returns_df[valid_tickers].copy().sort_index()
        metadata = {
            'analysis_mode': analysis_mode,
            'valid_tickers': valid_tickers,
            'requested_assets': len(valid_tickers),
        }

        if analysis_mode == 'strict':
            working_returns = working_returns.dropna(how='any')
            metadata['window_rule'] = 'all_assets_common_history'
        else:
            working_returns = working_returns.loc[working_returns.notna().any(axis=1)]
            metadata['window_rule'] = 'proxy_filled_history' if analysis_mode == 'proxy' else 'available_assets_by_month'

        if working_returns.empty:
            metadata['months'] = 0
            metadata['active_asset_min'] = 0
            metadata['active_asset_max'] = 0
            metadata['start_date'] = None
            metadata['end_date'] = None
            return pd.Series(dtype=float), norm_weights, metadata

        port_ret_list = []
        port_dates = []
        active_counts = []

        if analysis_mode == 'strict':
            current_weights = pd.Series(norm_weights, dtype=float)
            for date, row in working_returns.iterrows():
                port_ret_list.append((current_weights * row).sum())
                port_dates.append(date)
                active_counts.append(int(row.notna().sum()))

                current_weights = current_weights * (1 + row)
                if current_weights.sum() > 0:
                    current_weights /= current_weights.sum()
                else:
                    current_weights = pd.Series(norm_weights, dtype=float)

                if PortfolioAnalyzer._should_rebalance(date, rebalance_freq):
                    current_weights = pd.Series(norm_weights, dtype=float)
        else:
            current_weights = pd.Series(0.0, index=valid_tickers, dtype=float)
            for date, row in working_returns.iterrows():
                available_assets = row.dropna().index.tolist()
                if not available_assets:
                    continue

                active_counts.append(len(available_assets))
                effective_weights = current_weights[available_assets]
                if effective_weights.sum() <= 0:
                    effective_weights = PortfolioAnalyzer._normalized_available_weights(norm_weights, available_assets)
                else:
                    effective_weights = effective_weights / effective_weights.sum()

                active_returns = row[available_assets]
                port_ret_list.append((effective_weights * active_returns).sum())
                port_dates.append(date)

                updated_weights = effective_weights * (1 + active_returns)
                if updated_weights.sum() > 0:
                    updated_weights = updated_weights / updated_weights.sum()
                else:
                    updated_weights = PortfolioAnalyzer._normalized_available_weights(norm_weights, available_assets)

                current_weights = pd.Series(0.0, index=valid_tickers, dtype=float)
                current_weights.loc[available_assets] = updated_weights

                if PortfolioAnalyzer._should_rebalance(date, rebalance_freq):
                    rebalanced = PortfolioAnalyzer._normalized_available_weights(norm_weights, available_assets)
                    current_weights = pd.Series(0.0, index=valid_tickers, dtype=float)
                    current_weights.loc[available_assets] = rebalanced

        port_series = pd.Series(port_ret_list, index=port_dates, dtype=float)
        metadata['months'] = len(port_series)
        metadata['active_asset_min'] = min(active_counts) if active_counts else 0
        metadata['active_asset_max'] = max(active_counts) if active_counts else 0
        metadata['start_date'] = str(port_series.index.min().date()) if not port_series.empty else None
        metadata['end_date'] = str(port_series.index.max().date()) if not port_series.empty else None
        return port_series, norm_weights, metadata

    @staticmethod
    def calculate_correlation_matrix(returns_df, min_periods=12):
        if returns_df.empty:
            return pd.DataFrame()
        return returns_df.corr(min_periods=min_periods)

    @staticmethod
    def perform_factor_regression(port_ret, factor_df, min_months=24):
        if port_ret.empty or factor_df is None or factor_df.empty:
            return None, None, 0

        df_y = port_ret.to_frame(name='y')
        df_x = factor_df.copy()
        common_idx = df_y.index.intersection(df_x.index)
        if common_idx.empty:
            return None, None, 0

        merged = pd.concat([df_y.loc[common_idx], df_x.loc[common_idx]], axis=1).dropna()
        sample_count = len(merged)
        if sample_count < min_months:
            return None, None, sample_count

        y = merged['y']
        X_cols = [c for c in merged.columns if c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        if not X_cols:
            return None, None, sample_count

        X = sm.add_constant(merged[X_cols])
        try:
            model = sm.OLS(y, X)
            results = model.fit()
            return results.params, results.rsquared, sample_count
        except:
            return None, None, sample_count

    @staticmethod
    def run_monte_carlo_simulation(port_ret, n_years=20, n_simulations=7500, initial_investment=1000000, random_seed=42):
        if port_ret.empty:
            return None, None

        n_months = n_years * 12
        
        clean_ret = port_ret.dropna()
        if clean_ret.empty:
            return None, None

        log_returns = np.log(1 + clean_ret)
        mu_log = log_returns.mean()    
        sigma_log = log_returns.std()  

        rng = np.random.default_rng(random_seed)
        simulated_log_returns = rng.normal(mu_log, sigma_log, size=(n_months, n_simulations))
        monthly_returns = np.exp(simulated_log_returns)
        
        price_paths = np.zeros((n_months + 1, n_simulations))
        price_paths[0] = initial_investment
        price_paths[1:] = initial_investment * np.cumprod(monthly_returns, axis=0)
        
        last_date = clean_ret.index[-1]
        future_dates = pd.date_range(start=last_date, periods=n_months + 1, freq='ME')
        
        percentiles = [10, 50, 90]
        stats_data = np.percentile(price_paths, percentiles, axis=1)
        df_stats = pd.DataFrame(stats_data.T, index=future_dates, columns=['p10', 'p50', 'p90'])
        
        final_values = price_paths[-1, :]
        
        return df_stats, final_values

    @staticmethod
    def calculate_calmar_ratio(port_ret):
        if port_ret.empty: return np.nan
        cum_ret = (1 + port_ret).cumprod()
        if len(port_ret) < 12: return np.nan
        cagr = (cum_ret.iloc[-1])**(12/len(port_ret)) - 1
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        if max_dd == 0: return np.nan
        return cagr / abs(max_dd)

    @staticmethod
    def calculate_omega_ratio(port_ret, threshold=0.0):
        if port_ret.empty: return np.nan
        gains = port_ret[port_ret > threshold] - threshold
        losses = threshold - port_ret[port_ret < threshold]
        sum_gains = gains.sum()
        sum_losses = losses.sum()
        if sum_losses == 0: return np.inf
        return sum_gains / sum_losses

    @staticmethod
    def calculate_information_ratio(port_ret, bench_ret):
        if port_ret.empty or bench_ret.empty: return np.nan, np.nan
        
        p_df = port_ret.to_frame(name='p')
        b_df = bench_ret.to_frame(name='b')
        
        p_df.index = p_df.index.to_period('M')
        b_df.index = b_df.index.to_period('M')
        
        common_idx = p_df.index.intersection(b_df.index)
        if common_idx.empty or len(common_idx) < 12: 
            return np.nan, np.nan
            
        merged = pd.concat([p_df.loc[common_idx], b_df.loc[common_idx]], axis=1).dropna()
        if len(merged) < 12: 
            return np.nan, np.nan
        
        cum_p = (1 + merged['p']).cumprod()
        cum_b = (1 + merged['b']).cumprod()
        
        years = len(merged) / 12
        cagr_p = (cum_p.iloc[-1]) ** (1 / years) - 1
        cagr_b = (cum_b.iloc[-1]) ** (1 / years) - 1
        
        annualized_active_return = cagr_p - cagr_b
        
        active_ret = merged['p'] - merged['b']
        tracking_error = active_ret.std() * np.sqrt(12)
        
        if tracking_error == 0: return np.nan, 0.0
        return annualized_active_return / tracking_error, tracking_error

    @staticmethod
    def perform_pca(returns_df):
        if returns_df.shape[1] < 2: 
            return 1.0, None
        
        pca = PCA(n_components=2)
        pca.fit(returns_df)
        
        loadings = pd.DataFrame(
            pca.components_.T, 
            index=returns_df.columns, 
            columns=['PC1', 'PC2']
        )
        
        return pca.explained_variance_ratio_[0], loadings

    @staticmethod
    def rolling_beta_analysis(port_ret, factor_df, window=24, min_months=24):
        if factor_df is None or factor_df.empty or port_ret.empty:
            return pd.DataFrame(), 0, None

        df_y = port_ret.to_frame(name='y')
        df_x = factor_df.copy()
        common_idx = df_y.index.intersection(df_x.index)
        if common_idx.empty:
            return pd.DataFrame(), 0, None

        merged = pd.concat([df_y.loc[common_idx], df_x.loc[common_idx]], axis=1).dropna()
        sample_count = len(merged)
        if sample_count < min_months:
            return pd.DataFrame(), sample_count, None

        y = merged['y']
        X = merged[[c for c in merged.columns if c != 'y']]

        actual_window = window
        if sample_count < window * 2:
            actual_window = max(12, sample_count // 2)
        if sample_count < actual_window or actual_window < 12:
            return pd.DataFrame(), sample_count, actual_window

        try:
            X_const = sm.add_constant(X)
            model = RollingOLS(y, X_const, window=actual_window)
            rres = model.fit()
            params = rres.params.copy()
            if 'const' in params.columns:
                params = params.drop(columns=['const'])
            return params.dropna(), sample_count, actual_window
        except:
            return pd.DataFrame(), sample_count, actual_window

    @staticmethod
    def cost_drag_simulation(port_ret, cost_tier):
        if port_ret.empty: return pd.Series(), pd.Series(), 0, 0
        
        cost_map = {'Low': 0.001, 'Medium': 0.006, 'High': 0.020}
        annual_cost = cost_map.get(cost_tier, 0.006)
        monthly_cost = (1 + annual_cost)**(1/12) - 1
        
        net_ret = port_ret - monthly_cost
        gross_cum = (1 + port_ret).cumprod()
        net_cum = (1 + net_ret).cumprod()
        
        final_gross = gross_cum.iloc[-1]
        final_net = net_cum.iloc[-1]
        
        diff_val = final_gross - final_net
        lost_pct = 1 - (final_net / final_gross) 
        
        return gross_cum, net_cum, diff_val, lost_pct

    @staticmethod
    def calculate_strict_attribution(returns_df, weights_dict):
        assets = list(weights_dict.keys())
        available_assets = [a for a in assets if a in returns_df.columns]
        if not available_assets: return pd.Series(dtype=float)
            
        w_series = pd.Series(weights_dict)
        total_w = w_series[available_assets].sum()
        initial_w = w_series[available_assets] / total_w
        
        r_df = returns_df[available_assets].copy()
        
        cum_r_index = (1 + r_df).cumprod()
        asset_values = cum_r_index.multiply(initial_w, axis=1)
        port_values = asset_values.sum(axis=1)
        
        weights_df = asset_values.div(port_values, axis=0).shift(1)
        weights_df.iloc[0] = initial_w
        
        port_ret = (weights_df * r_df).sum(axis=1)
        total_cum_ret = (1 + port_ret).prod() - 1
        
        log_return = np.log(1 + total_cum_ret)
        k = log_return / total_cum_ret if total_cum_ret != 0 else 1.0
            
        kt = np.log(1 + port_ret) / port_ret
        kt = kt.fillna(1.0)
        
        term = weights_df * r_df
        smoothed_term = term.multiply(kt, axis=0)
        
        final_attribution = smoothed_term.sum() / k
        
        return final_attribution.sort_values(ascending=True)

    @staticmethod
    def calculate_risk_contribution(returns_df, weights_dict):
        assets = list(weights_dict.keys())
        valid_assets = [a for a in assets if a in returns_df.columns]
        if not valid_assets:
            return pd.Series(dtype=float)

        w_series = pd.Series({k: weights_dict[k] for k in valid_assets})
        w_series = w_series / w_series.sum() 
        cov_matrix = returns_df[valid_assets].cov() * 12 
        port_vol = np.sqrt(w_series.T @ cov_matrix @ w_series)
        mrc = cov_matrix @ w_series / port_vol
        rc = w_series * mrc
        rc_pct = rc / port_vol
        return rc_pct

    @staticmethod
    def calculate_label_offsets(values, min_dist=0.08, base_y=1.05):
        if not values: return []
        indexed_values = sorted(enumerate(values), key=lambda x: x[1])
        y_offsets = [base_y] * len(values)
        val_range = max(values) - min(values)
        if val_range == 0: val_range = 1.0
        
        levels = [base_y] * len(values)
        current_level = base_y
        
        for i in range(1, len(indexed_values)):
            curr_val = indexed_values[i][1]
            prev_val = indexed_values[i-1][1]
            dist = (curr_val - prev_val) / val_range
            
            if dist < min_dist:
                prev_level = levels[i-1]
                if prev_level == base_y:
                    current_level = base_y + 0.15
                elif prev_level == base_y + 0.15:
                    current_level = base_y + 0.3
                else:
                    current_level = base_y
            else:
                current_level = base_y
            
            levels[i] = current_level
            
        final_offsets = [0.0] * len(values)
        for i, (orig_idx, _) in enumerate(indexed_values):
            final_offsets[orig_idx] = levels[i]
            
        return final_offsets

class PortfolioDiagnosticEngine:
    @staticmethod
    def generate_report(weights_dict, pca_ratio, port_ret, benchmark_ret=None, lang='ja'):
        report = {
            "type": "",
            "risk_comment": "",
            "diversification_comment": "",
            "action_plan": ""
        }
        
        num_assets = len(weights_dict)
        
        if num_assets == 1:
            report["type"] = get_text('diag_sniper_type', lang)
            report["diversification_comment"] = get_text('diag_sniper_div', lang)
            report["risk_comment"] = get_text('diag_sniper_risk', lang)
            report["action_plan"] = get_text('diag_sniper_act', lang)
        else:
            if pca_ratio >= 0.85:
                report["type"] = get_text('diag_fake_type', lang)
                report["diversification_comment"] = get_text('diag_fake_div', lang).format(pca_ratio=pca_ratio*100)
                report["risk_comment"] = get_text('diag_fake_risk', lang)
                report["action_plan"] = get_text('diag_fake_act', lang)
            elif pca_ratio <= 0.60:
                report["type"] = get_text('diag_fortress_type', lang)
                report["diversification_comment"] = get_text('diag_fortress_div', lang).format(pca_ratio=pca_ratio*100)
                report["risk_comment"] = get_text('diag_fortress_risk', lang)
                report["action_plan"] = get_text('diag_fortress_act', lang)
            else:
                report["type"] = get_text('diag_balanced_type', lang)
                report["diversification_comment"] = get_text('diag_balanced_div', lang).format(pca_ratio=pca_ratio*100)
                report["risk_comment"] = get_text('diag_balanced_risk', lang)
                report["action_plan"] = get_text('diag_balanced_act', lang)

        return report

    @staticmethod
    def get_skew_kurt_desc(port_ret, lang='ja'):
        if port_ret.empty: 
            return get_text('no_data', lang)
            
        skew = port_ret.skew()
        kurt = port_ret.kurt()
        desc = []
        
        if str(lang).upper() == 'JA':
            if skew < -0.5: desc.append("⚠️ 負の歪度: 通常時は安定していますが、稀に大きな急落が起きるリスクがあります（コツコツドカン型）。")
            elif skew > 0.5: desc.append("✅ 正の歪度: 損失は限定的ですが、稀に大きな利益が出る可能性があります。")
            if kurt > 2.0: desc.append("⚠️ ファットテール: 正規分布に比べて「極端な事象（暴騰・暴落）」が発生する確率が高い状態です。")
            return " ".join(desc) if desc else "統計的に標準的な分布（正規分布に近い）です。"
        else:
            if skew < -0.5: desc.append("⚠️ Negative Skewness: Normally stable, but risks sudden sharp drops.")
            elif skew > 0.5: desc.append("✅ Positive Skewness: Limited losses with potential for rare large gains.")
            if kurt > 2.0: desc.append("⚠️ Fat Tail: Higher probability of 'extreme events' (crashes/spikes) than a normal distribution.")
            return " ".join(desc) if desc else "Statistically normal distribution."

    @staticmethod
    def generate_factor_report(params, lang='ja'):
        if params is None: 
            return get_text('no_data', lang)
        
        comments = []
        
        hml = params.get('HML', 0)
        smb = params.get('SMB', 0)
        mkt = params.get('Mkt-RF', 1.0)
        rmw = params.get('RMW', 0)
        cma = params.get('CMA', 0)
        
        if hml > 0.15: comments.append(get_text('factor_hml_val', lang))
        elif hml < -0.15: comments.append(get_text('factor_hml_gro', lang))
        else: comments.append(get_text('factor_hml_neu', lang))
        
        if smb > 0.15: comments.append(get_text('factor_smb_sma', lang))
        elif smb < -0.15: comments.append(get_text('factor_smb_lar', lang))
        
        if mkt > 1.1: comments.append(get_text('factor_mkt_high', lang))
        elif mkt < 0.9: comments.append(get_text('factor_mkt_low', lang))
        
        if 'RMW' in params.index:
            if rmw > 0.15: comments.append(get_text('factor_rmw_high', lang))
            elif rmw < -0.15: comments.append(get_text('factor_rmw_low', lang))
            
        if 'CMA' in params.index:
            if cma > 0.15: comments.append(get_text('factor_cma_high', lang))
            elif cma < -0.15: comments.append(get_text('factor_cma_low', lang))

        return "\n".join(comments)
