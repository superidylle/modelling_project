# -*- coding: utf-8 -*-

from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from datetime import datetime
import performance_statistics as perf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
import config


class TearsheetStatistic():

    def __init__(self, data, title=None,  periods=252):
        self.title = ''.join(title)
        self.data = data
        self.periods = periods

    def get_result(self):

        return_s = pd.Series(self.data['return'])
        acum_rtn = pd.Series(self.data['acum_rtn']) + 1
        rolling = return_s.rolling(window=self.periods)
        rolling_sharpe_s = np.sqrt(self.periods) * (rolling.mean() / rolling.std())
        dd_s, max_dd, dd_dur = perf.create_drawdowns(self.data['acum_rtn'])

        statistics = {}
        statistics["sharpe"] = perf.create_sharpe_ratio(return_s, self.periods)

        statistics["drawdowns"] = dd_s
        statistics["max_drawdown"] = max_dd
        statistics["max_drawdown_pct"] = max_dd
        statistics["max_drawdown_duration"] = dd_dur
        statistics['acum_rtn'] = acum_rtn
        statistics['return_s'] = return_s
        statistics["rolling_sharpe"] = rolling_sharpe_s

        equity_b = pd.Series(self.data['SPY']).sort_index()
        return_b = equity_b.pct_change().fillna(0.0)
        acum_rtn_b = np.cumsum(return_b).fillna(method='pad') + 1

        dd_b, max_dd_b, dd_dur_b = perf.create_drawdowns(acum_rtn_b)
        statistics["sharpe_b"] = perf.create_sharpe_ratio(return_b)
        statistics["drawdowns_b"] = dd_b
        statistics["max_drawdown_pct_b"] = max_dd_b
        statistics["max_drawdown_duration_b"] = dd_dur_b
        statistics["returns_b"] = return_b
        statistics["acum_rtn_b"] = acum_rtn_b

        return statistics

    def _plot_equity(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.2f%%' % x

        equity = stats['acum_rtn'] * 100

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        benchmark = stats['acum_rtn_b'] * 100
        benchmark.plot(lw=2, color='gray', label='SPY', alpha=0.60, ax=ax, **kwargs)

        equity.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.set_ylabel('acum_rtn')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        return ax

    def _plot_drawdown(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        drawdown = stats['drawdowns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        underwater = -100 * drawdown
        underwater.plot(ax=ax, lw=2, kind='area', color='red', alpha=0.3, **kwargs)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
        ax.set_title('Drawdown (%)', fontweight='bold')
        return ax

    def _plot_monthly_returns(self, stats, ax=None, **kwargs):

        returns = stats['return_s']
        if ax is None:
            ax = plt.gca()

        monthly_ret = perf.aggregate_returns(returns, 'monthly')
        monthly_ret = monthly_ret.unstack()
        monthly_ret = np.round(monthly_ret, 3)
        monthly_ret.rename(
            columns={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                     5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                     9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
            inplace=True
        )

        sns.heatmap(
            monthly_ret.fillna(0) * 100.0,
            annot=True,
            fmt="0.1f",
            annot_kws={"size": 8},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=cm.RdYlGn,
            ax=ax, **kwargs)
        ax.set_title('Monthly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('')

        return ax

    def _plot_yearly_returns(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats['return_s']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')

        yly_ret = perf.aggregate_returns(returns, 'yearly') * 100.0
        yly_ret.plot(ax=ax, kind="bar")
        ax.set_title('Yearly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.grid(False)

        return ax

    def _plot_txt_curve(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats["return_s"]
        acum_rtn = stats['acum_rtn']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        tot_ret = acum_rtn[-1] - 1.0
        cagr = perf.create_cagr(acum_rtn, self.periods)
        sharpe = perf.create_sharpe_ratio(returns, self.periods)
        sortino = perf.create_sortino_ratio(returns, self.periods)
        rsq = perf.rsquared(range(acum_rtn.shape[0]), acum_rtn)
        dd, dd_max, dd_dur = perf.create_drawdowns(acum_rtn)

        ax.text(0.25, 7.9, 'Total Return', fontsize=8)
        ax.text(7.50, 7.9, '{:.0%}'.format(tot_ret), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 6.9, 'CAGR', fontsize=8)
        ax.text(7.50, 6.9, '{:.2%}'.format(cagr), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 5.9, 'Sharpe Ratio', fontsize=8)
        ax.text(7.50, 5.9, '{:.2f}'.format(sharpe), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 4.9, 'Sortino Ratio', fontsize=8)
        ax.text(7.50, 4.9, '{:.2f}'.format(sortino), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 3.9, 'Annual Volatility', fontsize=8)
        ax.text(7.50, 3.9, '{:.2%}'.format(returns.std() * np.sqrt(252)), fontweight='bold',
                horizontalalignment='right', fontsize=8)

        ax.text(0.25, 2.9, 'R-Squared', fontsize=8)
        ax.text(7.50, 2.9, '{:.2f}'.format(rsq), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 1.9, 'Max Daily Drawdown', fontsize=8)
        ax.text(7.50, 1.9, '{:.2%}'.format(dd_max), color='red', fontweight='bold', horizontalalignment='right',
                fontsize=8)

        ax.text(0.25, 0.9, 'Max Drawdown Duration', fontsize=8)
        ax.text(7.50, 0.9, '{:.0f}'.format(dd_dur), fontweight='bold', horizontalalignment='right', fontsize=8)


        ax.set_title('Portfolio Performance', fontweight='bold')

        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax



    def _plot_txt_benchmark(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        return_b = stats["returns_b"]
        acum_rtn_b = stats["acum_rtn_b"]

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        tot_ret_b = acum_rtn_b[-1] - 1.0
        cagr_b = perf.create_cagr(acum_rtn_b, self.periods)
        sharpe_b = perf.create_sharpe_ratio(return_b, self.periods)
        sortino_b = perf.create_sortino_ratio(return_b, self.periods)
        rsq_b = perf.rsquared(range(acum_rtn_b.shape[0]), acum_rtn_b)
        dd_b, dd_max_b, dd_dur_b = perf.create_drawdowns(acum_rtn_b)

        ax.text(0.25, 7.9, 'Total Return', fontsize=8)
        ax.text(7.50, 7.9, '{:.0%}'.format(tot_ret_b), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 6.9, 'CAGR', fontsize=8)
        ax.text(7.50, 6.9, '{:.2%}'.format(cagr_b), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 5.9, 'Sharpe Ratio', fontsize=8)
        ax.text(7.50, 5.9, '{:.2f}'.format(sharpe_b), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 4.9, 'Sortino Ratio', fontsize=8)
        ax.text(7.50, 4.9, '{:.2f}'.format(sortino_b), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 3.9, 'Annual Volatility', fontsize=8)
        ax.text(7.50, 3.9, '{:.2%}'.format(return_b.std() * np.sqrt(252)), fontweight='bold',
                horizontalalignment='right', fontsize=8)

        ax.text(0.25, 2.9, 'R-Squared', fontsize=8)
        ax.text(7.50, 2.9, '{:.2f}'.format(rsq_b), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 1.9, 'Max Daily Drawdown', fontsize=8)
        ax.text(7.50, 1.9, '{:.2%}'.format(dd_max_b), color='red', fontweight='bold', horizontalalignment='right',
                fontsize=8)

        ax.text(0.25, 0.9, 'Max Drawdown Duration', fontsize=8)
        ax.text(7.50, 0.9, '{:.0f}'.format(dd_dur_b), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.set_title('Benchmark Performance', fontweight='bold')

        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax


    def _plot_txt_time(self, stats, ax=None, **kwargs):
        """
        Outputs the statistics for various time frames.
        """
        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats['return_s']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        mly_ret = perf.aggregate_returns(returns, 'monthly')
        yly_ret = perf.aggregate_returns(returns, 'yearly')

        mly_pct = mly_ret[mly_ret >= 0].shape[0] / float(mly_ret.shape[0])
        mly_avg_win_pct = np.mean(mly_ret[mly_ret >= 0])
        mly_avg_loss_pct = np.mean(mly_ret[mly_ret < 0])
        mly_max_win_pct = np.max(mly_ret)
        mly_max_loss_pct = np.min(mly_ret)
        yly_pct = yly_ret[yly_ret >= 0].shape[0] / float(yly_ret.shape[0])
        yly_max_win_pct = np.max(yly_ret)
        yly_max_loss_pct = np.min(yly_ret)

        ax.text(0.5, 8.9, 'Winning Months %', fontsize=8)
        ax.text(9.5, 8.9, '{:.0%}'.format(mly_pct), fontsize=8, fontweight='bold',
                horizontalalignment='right')

        ax.text(0.5, 7.9, 'Average Winning Month %', fontsize=8)
        ax.text(9.5, 7.9, '{:.2%}'.format(mly_avg_win_pct), fontsize=8, fontweight='bold',
                color='red' if mly_avg_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 6.9, 'Average Losing Month %', fontsize=8)
        ax.text(9.5, 6.9, '{:.2%}'.format(mly_avg_loss_pct), fontsize=8, fontweight='bold',
                color='red' if mly_avg_loss_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 5.9, 'Best Month %', fontsize=8)
        ax.text(9.5, 5.9, '{:.2%}'.format(mly_max_win_pct), fontsize=8, fontweight='bold',
                color='red' if mly_max_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 4.9, 'Worst Month %', fontsize=8)
        ax.text(9.5, 4.9, '{:.2%}'.format(mly_max_loss_pct), fontsize=8, fontweight='bold',
                color='red' if mly_max_loss_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 3.9, 'Winning Years %', fontsize=8)
        ax.text(9.5, 3.9, '{:.0%}'.format(yly_pct), fontsize=8, fontweight='bold',
                horizontalalignment='right')

        ax.text(0.5, 2.9, 'Best Year %', fontsize=8)
        ax.text(9.5, 2.9, '{:.2%}'.format(yly_max_win_pct), fontsize=8,
                fontweight='bold', color='red' if yly_max_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 1.9, 'Worst Year %', fontsize=8)
        ax.text(9.5, 1.9, '{:.2%}'.format(yly_max_loss_pct), fontsize=8,
                fontweight='bold', color='red' if yly_max_loss_pct < 0 else 'green',
                horizontalalignment='right')

        # ax.text(0.5, 0.9, 'Positive 12 Month Periods', fontsize=8)
        # ax.text(9.5, 0.9, num_trades, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.set_title('Time', fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax


    def plot_results(self, filename=None):

        rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.995',
            'figure.facecolor': '0.97',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.labelweight': 'bold',
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        }
        sns.set_context(rc)
        sns.set_style("whitegrid")
        sns.set_palette("deep", desat=.6)

        vertical_sections = 5
        fig = plt.figure(figsize=(10, vertical_sections * 3.5))
        fig.suptitle(self.title, y=0.94, weight='bold')
        # fig.suptitle(self.title, weight='bold')

        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.5)

        stats = self.get_result()
        ax_equity = plt.subplot(gs[:2, :])
        ax_drawdown = plt.subplot(gs[2, :])
        ax_monthly_returns = plt.subplot(gs[3, :2])
        ax_yearly_returns = plt.subplot(gs[3, 2])
        ax_txt_curve = plt.subplot(gs[4, 0])
        ax_txt_benchamark = plt.subplot(gs[4, 1])
        ax_txt_time = plt.subplot(gs[4, 2])

        self._plot_equity(stats, ax=ax_equity)
        self._plot_drawdown(stats, ax=ax_drawdown)
        self._plot_monthly_returns(stats, ax=ax_monthly_returns)
        self._plot_yearly_returns(stats, ax=ax_yearly_returns)
        self._plot_txt_curve(stats, ax=ax_txt_curve)
        self._plot_txt_benchmark(stats, ax=ax_txt_benchamark)
        self._plot_txt_time(stats, ax=ax_txt_time)

        plt.show()

        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches='tight')

    def get_filename(self, filename=""):
        if filename == "":
            now = datetime.utcnow()
            filename = "tearsheet_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            filename = config.output_data_path + '/' + filename
        return filename

    def save(self, filename=""):
        filename = self.get_filename(filename)
        self.plot_results(filename)




