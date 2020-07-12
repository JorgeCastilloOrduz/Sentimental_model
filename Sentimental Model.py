import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.data.factset.ownership import Form3AggregatedTrades
from quantopian.pipeline.data.factset.ownership import Form4and5AggregatedTrades
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline

from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals

import quantopian.pipeline.data.factset.estimates as fe
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.domain import US_EQUITIES

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 400

# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 1.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 1.0 / TOTAL_POSITIONS


def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.

    Use this function for initializing state or other bookkeeping.

    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in
        your algorithm. context is automatically passed to initialize,
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information
        about current positions.
    """

    algo.attach_pipeline(
        make_pipeline(),
        'data_pipe'
    )

    # Attach the pipeline for the risk model factors that we
    # want to neutralize in the optimization step. The 'risk_factors' string is
    # used to retrieve the output of the pipeline in before_trading_start below.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)
def insiders():

    qtu = QTradableStocksUS()
    insider_txns_form3_90d = Form3AggregatedTrades.slice(False, 90)
    insider_txns_form4and5_90d = Form4and5AggregatedTrades.slice(False, 90)
    unique_filers_form3_90d = insider_txns_form3_90d.num_unique_filers.latest
    unique_buyers_form4and5_90d = insider_txns_form4and5_90d.num_unique_buyers.latest
    unique_sellers_form4and5_90d = insider_txns_form4and5_90d.num_unique_sellers.latest
    # Sum the unique buyers from each form together.
    unique_buyers_90d = unique_filers_form3_90d + unique_buyers_form4and5_90d
    unique_sellers_90d = unique_sellers_form4and5_90d
    # Compute the fractions of insiders buying and selling.
    frac_insiders_buying_90d = unique_buyers_90d / (unique_buyers_90d + unique_sellers_90d)
    frac_insiders_selling_90d = unique_sellers_90d / (unique_buyers_90d + unique_sellers_90d)

    # compute factor as buying-selling rank zscores
    alpha_factor = frac_insiders_buying_90d - frac_insiders_selling_90d
    alpha_winsorized = alpha_factor.winsorize(min_percentile=0.10,
                                              max_percentile=0.90
                                              )
    alpha = alpha_winsorized.rank().zscore()

    screen = qtu & ~alpha_factor.isnull() & alpha_factor.isfinite()

    return alpha, screen

def public_opinion():
    qtu = QTradableStocksUS()
    sentiment_stocktwits = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=2,
    )
    sentiment_news = SimpleMovingAverage(
        inputs=[sentiment.sentiment_signal],
        window_length=2
    )
    sentiment_factor = sentiment_stocktwits + sentiment_news
    sentiment_factor_winsorized = sentiment_factor.winsorize(
        min_percentile=0.10,
        max_percentile=0.90)

    sentiment_factor_rank =  sentiment_factor_winsorized.rank().zscore()

    screen = qtu & ~sentiment_stocktwits.isnull() & ~sentiment_news.isnull() & sentiment_factor.isfinite()

    return sentiment_factor_rank, screen

def up_and_down():
    qtu = QTradableStocksUS()

    fq1_eps_cons = fe.PeriodicConsensus.slice('EPS', 'qf', 1)
    fq1_eps_cons_up = fq1_eps_cons.up.latest
    fq1_eps_cons_down = fq1_eps_cons.down.latest

    alpha_factor = fq1_eps_cons_up - fq1_eps_cons_down
    alpha_winsorized = alpha_factor.winsorize(min_percentile=0.10,
                                              max_percentile=0.90)

    alpha_rank = alpha_winsorized.rank().zscore()

    screen = qtu & ~alpha_factor.isnull() & alpha_factor.isfinite()

    return alpha_rank, screen

def make_pipeline():
    insiders_factor, screen_insiders = insiders()
    up_and_down_factor, screen_up_and_down = up_and_down()
    sentiment_factor, screen_sentiment = public_opinion()

    combined_factor =  up_and_down_factor + 2*sentiment_factor + insiders_factor

    return Pipeline(columns={'alpha_factor': combined_factor},
                    screen = screen_up_and_down & screen_sentiment&screen_insiders, domain=US_EQUITIES)



def before_trading_start(context, data):
    """
    Optional core function called automatically before the open of each market day.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        An object that provides methods to get price and volume data, check
        whether a security exists, and check the last time a security traded.
    """
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('data_pipe')

    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):
    """
    A function scheduled to run every day at market close in order to record
    strategy information.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    """
    A function scheduled to run once every Monday at 10AM ET in order to
    rebalance the longs and shorts lists.

    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings

    # Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.alpha_factor)

    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())
    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    #constraints.append(neutralize_risk_factors)

    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
