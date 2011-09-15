//
// This is an automated-generated file. Please do not edit it manually.
// Use X-ray to edit metadata and "tbstrategy export" instead.
//
#ifndef __TBRICKS_DEFINITIONS_H__
#define __TBRICKS_DEFINITIONS_H__

#include "strategy/API.h"

namespace tbricks {

namespace plug_ins {

// Plug-in "EQ_T_V2", description "EQ_T_V2"
tbricks::Identifier EQ_T_V2();
// Plug-in "TheoSpreader", description ""
tbricks::Identifier TheoSpreader();
// Plug-in "TheoQTParent", description ""
tbricks::Identifier TheoQTParent();
// Plug-in "MarketReplayTDF", description "Replays the BBO and the trade statistics of the instrument specified (and previously captured through SignalsTickCapture strategy) to the All Options Internal Market"
tbricks::Identifier MarketReplayTDF();
// Plug-in "AutoQuoter2", description "Basic multi-venue and multi-ccy quoter"
tbricks::Identifier AutoQuoter2();
// Plug-in "STEB_V4", description "FX Equity Spreader"
tbricks::Identifier STEB_V4();
// Plug-in "FXspreader", description "Basic multi-venue and multi-ccy quoter"
tbricks::Identifier FXspreader();
// Plug-in "OpeningAuction", description "Opening auction correlation strategy"
tbricks::Identifier OpeningAuction();
// Plug-in "PositionReport", description "Aggregated views of positions per portfolio"
tbricks::Identifier PositionReport();
// Plug-in "StreamChecker", description ""
tbricks::Identifier StreamChecker();
// Plug-in "ExposureAndFees", description ""
tbricks::Identifier ExposureAndFees();
// Plug-in "SetFutureFrontMonthClose", description ""
tbricks::Identifier SetFutureFrontMonthClose();
// Plug-in "BestPriceLogger", description ""
tbricks::Identifier BestPriceLogger();
// Plug-in "TradeFrame.2", description ""
tbricks::Identifier TradeFrame_46_2();
// Plug-in "TradeFrame", description ""
tbricks::Identifier TradeFrame();
// Plug-in "SG1_V5", description ""
tbricks::Identifier SG1V5();
// Plug-in "PortfolioTrader_V1", description "trade portfolio based on statistics"
tbricks::Identifier PortfolioTraderV1();
// Plug-in "Indicators", description ""
tbricks::Identifier Indicators();
// Plug-in "WeightHedger_V2", description ""
tbricks::Identifier WeightHedgerV2();
// Plug-in "VWAP-1", description ""
tbricks::Identifier VWAP_1();
// Plug-in "GeneralLogger", description ""
tbricks::Identifier GeneralLogger();
// Plug-in "DailyPnl", description ""
tbricks::Identifier DailyPnl();
// Plug-in "STEB_V1", description "MC "
tbricks::Identifier STEB_V1();
// Plug-in "SG1_MM", description ""
tbricks::Identifier SG1MM();
// Plug-in "EquityQuoter", description "EquityQuoter strategy."
tbricks::Identifier EquityQuoter();
// Plug-in "AutoQuoter", description "Basic multi-venue and multi-ccy quoter"
tbricks::Identifier AutoQuoter();
// Plug-in "PublicTradesLogger", description ""
tbricks::Identifier PublicTradesLogger();
// Plug-in "STEB_V5", description ""
tbricks::Identifier STEB_V5();
// Plug-in "AO_MarketReflector", description "New Market Reflector"
tbricks::Identifier AO_MarketReflector();
// Plug-in "STEB_V3", description "FX Equity Cluster"
tbricks::Identifier STEB_V3();
// Plug-in "TheoQTA", description ""
tbricks::Identifier TheoQTA();
// Plug-in "TestPluginDemo", description "Test plugin demo"
tbricks::Identifier TestPluginDemo();
// Plug-in "PublicTradesMonitor", description ""
tbricks::Identifier PublicTradesMonitor();
// Plug-in "TickCapture", description ""
tbricks::Identifier TickCapture();
}

// DurationType
enum DurationType
{
    // Item 0
    DurationTypeSec = 0,
    // Item 1
    DurationTypeTick = 1,
    // Item 2
    DurationTypeTickMin = 2
};

// Theo Method Ref Enum
enum TheoMethodRefEnum
{
    // Mid EMA
    TheoMethodRefEnumMidEMA = 0,
    // Sliding VWAP
    TheoMethodRefEnumSlidingVWAP = 1,
    // Spread EMA
    TheoMethodRefEnumSpreadEMA = 2
};

// FeeMICs
enum FeeMICs
{
    // Item 0
    FeeMICsAFET = 0,
    // Item 1
    FeeMICsALTX = 1,
    // Item 2
    FeeMICsBATS = 2,
    // Item 3
    FeeMICsCHIX = 3,
    // Item 4
    FeeMICsNURO = 4,
    // Item 5
    FeeMICsXETRA = 5
};

// EODRowAggregation
enum EODRowAggregation
{
    // Item 0
    EODRowAggregationVenue_AND_MIC = 0,
    // Item 2
    EODRowAggregationISIN = 1,
    // Item 1
    EODRowAggregationIVID = 2,
    // Item 3
    EODRowAggregationCurrency = 3
};

// SignalGenerator
enum SignalGenerator
{
    // SG1
    SignalGeneratorSG1 = 0,
    // SG2
    SignalGeneratorSG2 = 1,
    // SG3
    SignalGeneratorSG3 = 2,
    // SG4
    SignalGeneratorSG4 = 3,
    // SG5
    SignalGeneratorSG5 = 4,
    // SG6
    SignalGeneratorSG6 = 5
};

// ClosePosAggressLevel
enum ClosePosAggressLevel
{
    // HitOnStopLoss
    ClosePosAggressLevelHitOnStopLoss = 0,
    // Join
    ClosePosAggressLevelJoin = 1,
    // Dime
    ClosePosAggressLevelDime = 2,
    // Hit
    ClosePosAggressLevelHit = 3
};

// ActionOnStopLossEnum
enum ActionOnStopLossEnum
{
    // ClosePosition
    ActionOnStopLossEnumClosePosition = 0,
    // Pause
    ActionOnStopLossEnumPause = 1
};

// Processed Data
enum ProcessedData
{
    // Item 0
    ProcessedDataTrade = 0,
    // Item 1
    ProcessedDataMidpoint = 1
};

// OpenPosAggressLevel
enum OpenPosAggressLevel
{
    // Join
    OpenPosAggressLevelJoin = 0,
    // Dime
    OpenPosAggressLevelDime = 1,
    // HitOnSpreadVal
    OpenPosAggressLevelHitOnSpreadVal = 2,
    // Hit
    OpenPosAggressLevelHit = 3
};

// FeeRowAggregation
enum FeeRowAggregation
{
    // Item 0
    FeeRowAggregationVenue_AND_MIC = 0,
    // Item 1
    FeeRowAggregationISIN = 1
};

// Trigger Signal Value
enum TriggerSignalValue
{
    // Item 0
    TriggerSignalValueLast = 0,
    // Item 1
    TriggerSignalValueClose = 1
};

// SGWindowTypeEnum
enum SGWindowTypeEnum
{
    // Undef
    SGWindowTypeEnumUndef = -1,
    // Full
    SGWindowTypeEnumFull = 0,
    // Volume
    SGWindowTypeEnumVolume = 1,
    // Ticks
    SGWindowTypeEnumTicks = 2,
    // TimeSec
    SGWindowTypeEnumTimeSec = 3
};

// StreamTypes
enum StreamTypes
{
    // BestPriceStream
    StreamTypesBestPriceStream = 0,
    // StatisticsStream
    StreamTypesStatisticsStream = 1,
    // DepthStream
    StreamTypesDepthStream = 2,
    // InstrumentStatusStream
    StreamTypesInstrumentStatusStream = 3,
    // InstrumentStream
    StreamTypesInstrumentStream = 4,
    // PositionStream
    StreamTypesPositionStream = 5,
    // PublicTradeStream
    StreamTypesPublicTradeStream = 6,
    // TradeStream
    StreamTypesTradeStream = 7
};

// Theo Method Enum
enum TheoMethodEnum
{
    // Item 0
    TheoMethodEnumBaseMid = 0,
    // Item 2
    TheoMethodEnumBaseMidEMA = 1,
    // Item 3
    TheoMethodEnumBaseLastEMA = 2
};

// Trading Mode
enum TradingMode
{
    // Item 0
    TradingModeTF = 0,
    // Item 1
    TradingModeMM = 1
};

// View Indicator
enum ViewIndicator
{
    // Item 0
    ViewIndicatorADP = 0,
    // Item 1
    ViewIndicatorEMA = 1,
    // Item 2
    ViewIndicatorADPMed = 2,
    // Item 3
    ViewIndicatorADPMean = 3,
    // Item 4
    ViewIndicatorADPMedTR = 4,
    // Item 5
    ViewIndicatorADPMedTRMP = 5,
    // Item 6
    ViewIndicatorADPMedTRsigma = 6,
    // Item 7
    ViewIndicatorADPMedTRMPsigma = 7,
    // Item 8
    ViewIndicatorADPMedTRsigmaCROSS = 8
};

namespace instrument_parameters {

// Instrument parameter "FDAXFrontMonthClose", type "Price"
tbricks::Uuid FDAXFrontMonthClose();

// Instrument parameter "FESXFrontMonthClose", type "Price"
tbricks::Uuid FESXFrontMonthClose();

// Instrument parameter "News", type "String"
tbricks::Uuid News();

// Instrument parameter "Hidden quoter", type "Boolean"
tbricks::Uuid HiddenQuoter();

// Instrument parameter "Dimer", type "Boolean"
tbricks::Uuid Dimer();

// Instrument parameter "Quote min volume", type "Volume"
tbricks::Uuid QuoteMinVolume();

// Instrument parameter "Quote hedge volume", type "Double"
tbricks::Uuid QuoteHedgeVolume();

// Instrument parameter "DaxWeight", type "Double"
tbricks::Uuid DaxWeight();

// Instrument parameter "ExchangeFees", type "Table"
tbricks::Uuid ExchangeFees();

// Instrument parameter "QuoterBid", type "Boolean"
tbricks::Uuid QuoterBid();

// Instrument parameter "Clearing CH", type "String"
tbricks::Uuid ClearingCH();

// Instrument parameter "Quote volume", type "Volume"
tbricks::Uuid QuoteVolume();

// Instrument parameter "NewsFactor", type "Integer"
tbricks::Uuid NewsFactor();

// Instrument parameter "Quote spread", type "Double"
tbricks::Uuid QuoteSpread();

// Instrument parameter "ClickTrader: Enable", type "Boolean"
tbricks::Uuid ClickTrader_58_Enable();

// Instrument parameter "Clearing NL", type "String"
tbricks::Uuid ClearingNL();

// Instrument parameter "QuoterAsk", type "Boolean"
tbricks::Uuid QuoterAsk();

// Instrument parameter "Fixed fee", type "Double"
tbricks::Uuid FixedFee();

}

namespace strategy_parameters {

// Strategy parameter "Credit Bid", type "Integer"
tbricks::ParameterDefinition CreditBid();

// Strategy parameter "Credit Ask", type "Integer"
tbricks::ParameterDefinition CreditAsk();

// Strategy parameter "Source Instrument", type "InstrumentIdentifier"
tbricks::ParameterDefinition SourceInstrument();

// Strategy parameter "Target Instrument", type "InstrumentIdentifier"
tbricks::ParameterDefinition TargetInstrument();

// Strategy parameter "Source Venue", type "VenueIdentifier"
tbricks::ParameterDefinition SourceVenue();

// Strategy parameter "Target Venue", type "VenueIdentifier"
tbricks::ParameterDefinition TargetVenue();

// Strategy parameter "Bid Quote Status", type "Integer"
tbricks::ParameterDefinition BidQuoteStatus();

// Strategy parameter "Ask Quote Status", type "Integer"
tbricks::ParameterDefinition AskQuoteStatus();

// Strategy parameter "Quote Bid Qty", type "Volume"
tbricks::ParameterDefinition QuoteBidQty();

// Strategy parameter "Quote Ask Qty", type "Volume"
tbricks::ParameterDefinition QuoteAskQty();

// Strategy parameter "PnL", type "Double"
tbricks::ParameterDefinition PnL();

// Strategy parameter "Position", type "Double"
tbricks::ParameterDefinition Position();

// Strategy parameter "Enable", type "Boolean"
tbricks::ParameterDefinition Enable();

// Strategy parameter "HedgeVenue", type "VenueIdentifier"
tbricks::ParameterDefinition HedgeVenue();

// Strategy parameter "Quote Update Threshold", type "Integer"
tbricks::ParameterDefinition QuoteUpdateThreshold();

// Strategy parameter "Min Hedge Qty", type "Volume"
tbricks::ParameterDefinition MinHedgeQty();

// Strategy parameter "Account", type "String"
tbricks::ParameterDefinition Account();

// Strategy parameter "OpenCloseCode", type "String"
tbricks::ParameterDefinition OpenCloseCode();

// Strategy parameter "Takeup Member ID", type "String"
tbricks::ParameterDefinition TakeupMemberID();

// Strategy parameter "Text1", type "String"
tbricks::ParameterDefinition Text1();

// Strategy parameter "Text2", type "String"
tbricks::ParameterDefinition Text2();

// Strategy parameter "Text3", type "String"
tbricks::ParameterDefinition Text3();

// Strategy parameter "Skip Reasonable Price Check", type "Boolean"
tbricks::ParameterDefinition SkipReasonablePriceCheck();

// Strategy parameter "HedgeBookDepthMx", type "Double"
tbricks::ParameterDefinition HedgeBookDepthMx();

// Strategy parameter "Extra Data Name", type "String"
tbricks::ParameterDefinition ExtraDataName();

// Strategy parameter "Extra Data Value", type "String"
tbricks::ParameterDefinition ExtraDataValue();

// Strategy parameter "Max Spread", type "Double"
tbricks::ParameterDefinition MaxSpread();

// Strategy parameter "Force Max Spread", type "Boolean"
tbricks::ParameterDefinition ForceMaxSpread();

// Strategy parameter "Max Trades Per Second", type "Integer"
tbricks::ParameterDefinition MaxTradesPerSecond();

// Strategy parameter "Instrument_1", type "InstrumentIdentifier"
tbricks::ParameterDefinition Instrument1();

// Strategy parameter "Venue_1", type "VenueIdentifier"
tbricks::ParameterDefinition Venue1();

// Strategy parameter "MIC_1", type "MIC"
tbricks::ParameterDefinition MIC_1();

// Strategy parameter "Exchange Account_1", type "String"
tbricks::ParameterDefinition ExchangeAccount1();

// Strategy parameter "ClearingAccount_1", type "String"
tbricks::ParameterDefinition ClearingAccount1();

// Strategy parameter "User Account_1", type "String"
tbricks::ParameterDefinition UserAccount1();

// Strategy parameter "Instrument_2", type "InstrumentIdentifier"
tbricks::ParameterDefinition Instrument2();

// Strategy parameter "Venue_2", type "VenueIdentifier"
tbricks::ParameterDefinition Venue2();

// Strategy parameter "MIC_2", type "MIC"
tbricks::ParameterDefinition MIC_2();

// Strategy parameter "Exchange Account_2", type "String"
tbricks::ParameterDefinition ExchangeAccount2();

// Strategy parameter "ClearingAccount_2", type "String"
tbricks::ParameterDefinition ClearingAccount2();

// Strategy parameter "User Account_2", type "String"
tbricks::ParameterDefinition UserAccount2();

// Strategy parameter "# of sent orders", type "Integer"
tbricks::ParameterDefinition _HASH_OfSentOrders();

// Strategy parameter "TradeCounter", type "Integer"
tbricks::ParameterDefinition TradeCounter();

// Strategy parameter "StrategyInstanceName", type "String"
tbricks::ParameterDefinition StrategyInstanceName();

// Strategy parameter "Quote Margin Ticks", type "Integer"
tbricks::ParameterDefinition QuoteMarginTicks();

// Strategy parameter "Realized Pnl", type "Double"
tbricks::ParameterDefinition RealizedPnl();

// Strategy parameter "Unrealized Pnl", type "Double"
tbricks::ParameterDefinition UnrealizedPnl();

// Strategy parameter "Order Count_1", type "Integer"
tbricks::ParameterDefinition OrderCount1();

// Strategy parameter "Order Count_2", type "Integer"
tbricks::ParameterDefinition OrderCount2();

// Strategy parameter "Trade Count_2", type "Integer"
tbricks::ParameterDefinition TradeCount2();

// Strategy parameter "Trade Count_1", type "Integer"
tbricks::ParameterDefinition TradeCount1();

// Strategy parameter "Reset Counters", type "Boolean"
tbricks::ParameterDefinition ResetCounters();

// Strategy parameter "Enable Quote Bid", type "Boolean"
tbricks::ParameterDefinition EnableQuoteBid();

// Strategy parameter "Enable Quote Ask", type "Boolean"
tbricks::ParameterDefinition EnableQuoteAsk();

// Strategy parameter "Stop Quoting Position", type "Volume"
tbricks::ParameterDefinition StopQuotingPosition();

// Strategy parameter "Retreat Ticks", type "Integer"
tbricks::ParameterDefinition RetreatTicks();

// Strategy parameter "Total Position", type "Double"
tbricks::ParameterDefinition TotalPosition();

// Strategy parameter "Total Position Average Price", type "Double"
tbricks::ParameterDefinition TotalPositionAveragePrice();

// Strategy parameter "Quantity", type "Volume"
tbricks::ParameterDefinition Quantity();

// Strategy parameter "Cash Margin", type "Double"
tbricks::ParameterDefinition CashMargin();

// Strategy parameter "Update Threshold Ticks", type "Integer"
tbricks::ParameterDefinition UpdateThresholdTicks();

// Strategy parameter "Price multiplier", type "Double"
tbricks::ParameterDefinition PriceMultiplier();

// Strategy parameter "BookDepthMx", type "Double"
tbricks::ParameterDefinition BookDepthMx();

// Strategy parameter "Max Position", type "Double"
tbricks::ParameterDefinition MaxPosition();

// Strategy parameter "Cutoff Ratio", type "Double"
tbricks::ParameterDefinition CutoffRatio();

// Strategy parameter "Quote Qty Scale", type "Double"
tbricks::ParameterDefinition QuoteQtyScale();

// Strategy parameter "Quote Price Scale", type "Double"
tbricks::ParameterDefinition QuotePriceScale();

// Strategy parameter "Quote Price Constant", type "Double"
tbricks::ParameterDefinition QuotePriceConstant();

// Strategy parameter "EODMtm", type "Double"
tbricks::ParameterDefinition EODMtm();

// Strategy parameter "CashMarginParam", type "Double"
tbricks::ParameterDefinition CashMarginParam();

// Strategy parameter "Offset1", type "Double"
tbricks::ParameterDefinition Offset1();

// Strategy parameter "SGCloseIndicatorInstrument", type "Price"
tbricks::ParameterDefinition SGCloseIndicatorInstrument();

// Strategy parameter "Spread ask margin", type "Double"
tbricks::ParameterDefinition SpreadAskMargin();

// Strategy parameter "DebugMode", type "Boolean"
tbricks::ParameterDefinition DebugMode();

// Strategy parameter "SGCorrelation", type "Double"
tbricks::ParameterDefinition SGCorrelation();

// Strategy parameter "BidStatus", type "Integer"
tbricks::ParameterDefinition BidStatus();

// Strategy parameter "EnableInspectorStatistics", type "Boolean"
tbricks::ParameterDefinition EnableInspectorStatistics();

// Strategy parameter "SGPeriod", type "Integer"
tbricks::ParameterDefinition SGPeriod();

// Strategy parameter "FutureClosingPrice", type "Price"
tbricks::ParameterDefinition FutureClosingPrice();

// Strategy parameter "Use BBO", type "Boolean"
tbricks::ParameterDefinition UseBBO();

// Strategy parameter "Limit_1", type "Double"
tbricks::ParameterDefinition Limit1();

// Strategy parameter "Time Window Size", type "Integer"
tbricks::ParameterDefinition TimeWindowSize();

// Strategy parameter "FeeRowAggregation", type "Integer"
tbricks::ParameterDefinition FeeRowAggregation();

// Strategy parameter "Adapt Local View", type "Boolean"
tbricks::ParameterDefinition AdaptLocalView();

// Strategy parameter "SGNominalOrderAmount", type "Double"
tbricks::ParameterDefinition SGNominalOrderAmount();

// Strategy parameter "SGViewOnUnderPerf", type "Boolean"
tbricks::ParameterDefinition SGViewOnUnderPerf();

// Strategy parameter "Date1", type "DateTime"
tbricks::ParameterDefinition Date1();

// Strategy parameter "Implied FX - ask", type "Price"
tbricks::ParameterDefinition ImpliedFX___Ask();

// Strategy parameter "roAskPrice", type "Price"
tbricks::ParameterDefinition RoAskPrice();

// Strategy parameter "HedgeTakeProfitTicks", type "Integer"
tbricks::ParameterDefinition HedgeTakeProfitTicks();

// Strategy parameter "X_ticks", type "Integer"
tbricks::ParameterDefinition X_Ticks();

// Strategy parameter "roBestAskPrice", type "Price"
tbricks::ParameterDefinition RoBestAskPrice();

// Strategy parameter "SGMoveTicks", type "Integer"
tbricks::ParameterDefinition SGMoveTicks();

// Strategy parameter "Clean Deltas Children", type "Boolean"
tbricks::ParameterDefinition CleanDeltasChildren();

// Strategy parameter "HedgeAbsCutoffTicks", type "Integer"
tbricks::ParameterDefinition HedgeAbsCutoffTicks();

// Strategy parameter "Spread bid margin", type "Double"
tbricks::ParameterDefinition SpreadBidMargin();

// Strategy parameter "FX - inverted", type "Boolean"
tbricks::ParameterDefinition FX___Inverted();

// Strategy parameter "ExchangeAccount_3", type "String"
tbricks::ParameterDefinition ExchangeAccount3();

// Strategy parameter "Clean Deltas", type "Boolean"
tbricks::ParameterDefinition CleanDeltas();

// Strategy parameter "Y_ticks", type "Integer"
tbricks::ParameterDefinition Y_Ticks();

// Strategy parameter "MaxSpreadTicksToCross", type "Integer"
tbricks::ParameterDefinition MaxSpreadTicksToCross();

// Strategy parameter "Stop Loss ", type "Double"
tbricks::ParameterDefinition StopLoss();

// Strategy parameter "EODReportTable", type "Table"
tbricks::ParameterDefinition EODReportTable();

// Strategy parameter "TickFrequency", type "Integer"
tbricks::ParameterDefinition TickFrequency();

// Strategy parameter "JoinQuote", type "Boolean"
tbricks::ParameterDefinition JoinQuote();

// Strategy parameter "DisableBats", type "Boolean"
tbricks::ParameterDefinition DisableBats();

// Strategy parameter "BaseCurrency", type "Currency"
tbricks::ParameterDefinition BaseCurrency();

// Strategy parameter "SGTriggerThreshold", type "Double"
tbricks::ParameterDefinition SGTriggerThreshold();

// Strategy parameter "TradeLimitPerSecond", type "Integer"
tbricks::ParameterDefinition TradeLimitPerSecond();

// Strategy parameter "Adjustment", type "Double"
tbricks::ParameterDefinition Adjustment();

// Strategy parameter "Trigger Signal Value", type "Integer"
tbricks::ParameterDefinition TriggerSignalValue();

// Strategy parameter "Trading Mode", type "Integer"
tbricks::ParameterDefinition TradingMode();

// Strategy parameter "CurrentView", type "String"
tbricks::ParameterDefinition CurrentView();

// Strategy parameter "FX - spot mid", type "Price"
tbricks::ParameterDefinition FX___SpotMid();

// Strategy parameter "HedgeSubCutoffTicks", type "Integer"
tbricks::ParameterDefinition HedgeSubCutoffTicks();

// Strategy parameter "EODRowAggregation", type "Integer"
tbricks::ParameterDefinition EODRowAggregation();

// Strategy parameter "TurnoverBaseNormal", type "Double"
tbricks::ParameterDefinition TurnoverBaseNormal();

// Strategy parameter "DisableNuro", type "Boolean"
tbricks::ParameterDefinition DisableNuro();

// Strategy parameter "Time Interval", type "Integer"
tbricks::ParameterDefinition TimeInterval();

// Strategy parameter "Correlation factor", type "Double"
tbricks::ParameterDefinition CorrelationFactor();

// Strategy parameter "SGHedgeOffsetPercentage", type "Double"
tbricks::ParameterDefinition SGHedgeOffsetPercentage();

// Strategy parameter "Implied spread - mid", type "Price"
tbricks::ParameterDefinition ImpliedSpreadMid();

// Strategy parameter "ExposureAndFeesTable", type "Table"
tbricks::ParameterDefinition ExposureAndFeesTable();

// Strategy parameter "Implied FX - bid", type "Price"
tbricks::ParameterDefinition ImpliedFX___Bid();

// Strategy parameter "Enable TRL Stop", type "Boolean"
tbricks::ParameterDefinition EnableTRL_Stop();

// Strategy parameter "EnableInspectorDepth", type "Boolean"
tbricks::ParameterDefinition EnableInspectorDepth();

// Strategy parameter "HedgeTrailingStopTicks", type "Integer"
tbricks::ParameterDefinition HedgeTrailingStopTicks();

// Strategy parameter "Trade Value Per Stock", type "Price"
tbricks::ParameterDefinition TradeValuePerStock();

// Strategy parameter "StreamType", type "Integer"
tbricks::ParameterDefinition StreamType();

// Strategy parameter "EnableInstrument_1", type "Boolean"
tbricks::ParameterDefinition EnableInstrument1();

// Strategy parameter "Max Volume", type "Volume"
tbricks::ParameterDefinition MaxVolume();

// Strategy parameter "Theo Method Ref", type "Integer"
tbricks::ParameterDefinition TheoMethodRef();

// Strategy parameter "ClosePositionAggressLevelParam", type "Integer"
tbricks::ParameterDefinition ClosePositionAggressLevelParam();

// Strategy parameter "EODVenue", type "VenueIdentifier"
tbricks::ParameterDefinition EODVenue();

// Strategy parameter "EODMIC", type "MIC"
tbricks::ParameterDefinition EODMIC();

// Strategy parameter "HedgeOffset", type "Integer"
tbricks::ParameterDefinition HedgeOffset();

// Strategy parameter "EnableShiftOnFillConf", type "Boolean"
tbricks::ParameterDefinition EnableShiftOnFillConf();

// Strategy parameter "SGCutoffPercentage", type "Double"
tbricks::ParameterDefinition SGCutoffPercentage();

// Strategy parameter "ExpandPositionThreshold", type "Volume"
tbricks::ParameterDefinition ExpandPositionThreshold();

// Strategy parameter "ExpandPositionOnBetterPriceTicks", type "Integer"
tbricks::ParameterDefinition ExpandPositionOnBetterPriceTicks();

// Strategy parameter "FXPosition", type "Double"
tbricks::ParameterDefinition FXPosition();

// Strategy parameter "DurationType", type "Integer"
tbricks::ParameterDefinition DurationType();

// Strategy parameter "EndTime", type "DateTime"
tbricks::ParameterDefinition EndTime();

// Strategy parameter "OpenPositionAggressLevelParam", type "Integer"
tbricks::ParameterDefinition OpenPositionAggressLevelParam();

// Strategy parameter "SGMoveFactor", type "Double"
tbricks::ParameterDefinition SGMoveFactor();

// Strategy parameter "ToDate", type "DateTime"
tbricks::ParameterDefinition ToDate();

// Strategy parameter "roBestAskVolume", type "Volume"
tbricks::ParameterDefinition RoBestAskVolume();

// Strategy parameter "TurnoverBaseFX", type "Double"
tbricks::ParameterDefinition TurnoverBaseFX();

// Strategy parameter "IndicatorValue_2", type "Double"
tbricks::ParameterDefinition IndicatorValue2();

// Strategy parameter "EODCcy", type "Currency"
tbricks::ParameterDefinition EODCcy();

// Strategy parameter "ActionOnStopLoss", type "Integer"
tbricks::ParameterDefinition ActionOnStopLoss();

// Strategy parameter "roInstrument", type "String"
tbricks::ParameterDefinition RoInstrument();

// Strategy parameter "Phase", type "String"
tbricks::ParameterDefinition Phase();

// Strategy parameter "roBidPrice", type "Price"
tbricks::ParameterDefinition RoBidPrice();

// Strategy parameter "SGTrendLimit", type "Double"
tbricks::ParameterDefinition SGTrendLimit();

// Strategy parameter "SGBeta", type "Double"
tbricks::ParameterDefinition SGBeta();

// Strategy parameter "SGTrailingTicks", type "Integer"
tbricks::ParameterDefinition SGTrailingTicks();

// Strategy parameter "StartHedgePos", type "Double"
tbricks::ParameterDefinition StartHedgePos();

// Strategy parameter "roBidVolume", type "Volume"
tbricks::ParameterDefinition RoBidVolume();

// Strategy parameter "TurnoverEUR", type "Double"
tbricks::ParameterDefinition TurnoverEUR();

// Strategy parameter "SGDeviationLimitPercentage", type "Double"
tbricks::ParameterDefinition SGDeviationLimitPercentage();

// Strategy parameter "Number Of Periods", type "Integer"
tbricks::ParameterDefinition NumberOfPeriods();

// Strategy parameter "EnableInstrument_2", type "Boolean"
tbricks::ParameterDefinition EnableInstrument2();

// Strategy parameter "FX - market implied", type "Price"
tbricks::ParameterDefinition FX___MarketImplied();

// Strategy parameter "SGCloseTradeInstrument", type "Price"
tbricks::ParameterDefinition SGCloseTradeInstrument();

// Strategy parameter "SGWindowType", type "Integer"
tbricks::ParameterDefinition SGWindowType();

// Strategy parameter "EnableInspectorBestPrice", type "Boolean"
tbricks::ParameterDefinition EnableInspectorBestPrice();

// Strategy parameter "EnableInspectorPublicTrades", type "Boolean"
tbricks::ParameterDefinition EnableInspectorPublicTrades();

// Strategy parameter "TradeLimitPerSecondChildren", type "Integer"
tbricks::ParameterDefinition TradeLimitPerSecondChildren();

// Strategy parameter "Price Shift_1", type "Double"
tbricks::ParameterDefinition PriceShift1();

// Strategy parameter "EODInstrumentName", type "String"
tbricks::ParameterDefinition EODInstrumentName();

// Strategy parameter "Processed Data", type "Integer"
tbricks::ParameterDefinition ProcessedData();

// Strategy parameter "Sensitivity", type "Double"
tbricks::ParameterDefinition Sensitivity();

// Strategy parameter "DisableChix", type "Boolean"
tbricks::ParameterDefinition DisableChix();

// Strategy parameter "HedgeDimeValue", type "Double"
tbricks::ParameterDefinition HedgeDimeValue();

// Strategy parameter "ClearingCost", type "Double"
tbricks::ParameterDefinition ClearingCost();

// Strategy parameter "Indicator", type "Integer"
tbricks::ParameterDefinition Indicator();

// Strategy parameter "UseMid_1", type "Boolean"
tbricks::ParameterDefinition UseMid1();

// Strategy parameter "RetreatMx_1", type "Double"
tbricks::ParameterDefinition RetreatMx1();

// Strategy parameter "ClosePositionInView", type "Integer"
tbricks::ParameterDefinition ClosePositionInView();

// Strategy parameter "Order Count_3", type "Integer"
tbricks::ParameterDefinition OrderCount3();

// Strategy parameter "StatusMessage", type "String"
tbricks::ParameterDefinition StatusMessage();

// Strategy parameter "Venue_3", type "VenueIdentifier"
tbricks::ParameterDefinition Venue3();

// Strategy parameter "EODIsinCode", type "String"
tbricks::ParameterDefinition EODIsinCode();

// Strategy parameter "Premium", type "Double"
tbricks::ParameterDefinition Premium();

// Strategy parameter "EODInstrument", type "String"
tbricks::ParameterDefinition EODInstrument();

// Strategy parameter "roPosition", type "Volume"
tbricks::ParameterDefinition RoPosition();

// Strategy parameter "roVenueTurnover", type "Volume"
tbricks::ParameterDefinition RoVenueTurnover();

// Strategy parameter "Trade Count_3", type "Integer"
tbricks::ParameterDefinition TradeCount3();

// Strategy parameter "QuoteBoth", type "Boolean"
tbricks::ParameterDefinition QuoteBoth();

// Strategy parameter "Quote For Closure Children", type "Boolean"
tbricks::ParameterDefinition QuoteForClosureChildren();

// Strategy parameter "ClearingAccount_3", type "String"
tbricks::ParameterDefinition ClearingAccount3();

// Strategy parameter "roMIC", type "MIC"
tbricks::ParameterDefinition RoMIC();

// Strategy parameter "QuoteCashValue", type "Double"
tbricks::ParameterDefinition QuoteCashValue();

// Strategy parameter "DisableXetra", type "Boolean"
tbricks::ParameterDefinition DisableXetra();

// Strategy parameter "SettlementCost", type "Double"
tbricks::ParameterDefinition SettlementCost();

// Strategy parameter "Hack Level", type "Double"
tbricks::ParameterDefinition HackLevel();

// Strategy parameter "Local View", type "Integer"
tbricks::ParameterDefinition LocalView();

// Strategy parameter "EquityQuoter Parameters", type "Table"
tbricks::ParameterDefinition EquityQuoterParameters();

// Strategy parameter "roBestBidPrice", type "Price"
tbricks::ParameterDefinition RoBestBidPrice();

// Strategy parameter "Enable Gap", type "Boolean"
tbricks::ParameterDefinition EnableGap();

// Strategy parameter "CashMarginAsk", type "Double"
tbricks::ParameterDefinition CashMarginAsk();

// Strategy parameter "DisableEuronext", type "Boolean"
tbricks::ParameterDefinition DisableEuronext();

// Strategy parameter "Enable Quote Bid Children", type "Boolean"
tbricks::ParameterDefinition EnableQuoteBidChildren();

// Strategy parameter "SGAlpha", type "Double"
tbricks::ParameterDefinition SGAlpha();

// Strategy parameter "CashInBaseCCy", type "Double"
tbricks::ParameterDefinition CashInBaseCCy();

// Strategy parameter "Number Of Active Orders", type "Integer"
tbricks::ParameterDefinition NumberOfActiveOrders();

// Strategy parameter "Period Duration", type "Integer"
tbricks::ParameterDefinition PeriodDuration();

// Strategy parameter "Retreat BPs", type "Integer"
tbricks::ParameterDefinition RetreatBPs();

// Strategy parameter "Enable Trailing Average", type "Boolean"
tbricks::ParameterDefinition EnableTrailingAverage();

// Strategy parameter "SGTimeSlice", type "Integer"
tbricks::ParameterDefinition SGTimeSlice();

// Strategy parameter "StreamDataMemberValue", type "String"
tbricks::ParameterDefinition StreamDataMemberValue();

// Strategy parameter "SGTrailFactor", type "Double"
tbricks::ParameterDefinition SGTrailFactor();

// Strategy parameter "EnableDetailedInformation", type "Boolean"
tbricks::ParameterDefinition EnableDetailedInformation();

// Strategy parameter "EODPnL", type "Double"
tbricks::ParameterDefinition EODPnL();

// Strategy parameter "Offset BPs", type "Integer"
tbricks::ParameterDefinition OffsetBPs();

// Strategy parameter "RebalanceCashValue", type "Double"
tbricks::ParameterDefinition RebalanceCashValue();

// Strategy parameter "CashMarginBid", type "Double"
tbricks::ParameterDefinition CashMarginBid();

// Strategy parameter "FXPositionAveragePrice", type "Double"
tbricks::ParameterDefinition FXPositionAveragePrice();

// Strategy parameter "Pause After Stop Loss Delta Clean", type "Boolean"
tbricks::ParameterDefinition PauseAfterStopLossDeltaClean();

// Strategy parameter "RebalanceOnCash", type "Boolean"
tbricks::ParameterDefinition RebalanceOnCash();

// Strategy parameter "SGWindowSize", type "Integer"
tbricks::ParameterDefinition SGWindowSize();

// Strategy parameter "ROI", type "Double"
tbricks::ParameterDefinition ROI();

// Strategy parameter "Adjust Bandwidth BPs", type "Integer"
tbricks::ParameterDefinition AdjustBandwidthBPs();

// Strategy parameter "FX - position realized", type "Price"
tbricks::ParameterDefinition FX___PositionRealized();

// Strategy parameter "Instrument_3", type "InstrumentIdentifier"
tbricks::ParameterDefinition Instrument3();

// Strategy parameter "TurnoverBase", type "Double"
tbricks::ParameterDefinition TurnoverBase();

// Strategy parameter "Enable Counter Order", type "Boolean"
tbricks::ParameterDefinition EnableCounterOrder();

// Strategy parameter "CurrencyExposure", type "Double"
tbricks::ParameterDefinition CurrencyExposure();

// Strategy parameter "Quote for closure", type "Boolean"
tbricks::ParameterDefinition QuoteForClosure();

// Strategy parameter "SGClaimOffsetPercentage", type "Double"
tbricks::ParameterDefinition SGClaimOffsetPercentage();

// Strategy parameter "roAvgPrice", type "Price"
tbricks::ParameterDefinition RoAvgPrice();

// Strategy parameter "IndicatorValue_4", type "Double"
tbricks::ParameterDefinition IndicatorValue4();

// Strategy parameter "roAskVolume", type "Volume"
tbricks::ParameterDefinition RoAskVolume();

// Strategy parameter "CashExtraMargin", type "Double"
tbricks::ParameterDefinition CashExtraMargin();

// Strategy parameter "TurnoverVol", type "Double"
tbricks::ParameterDefinition TurnoverVol();

// Strategy parameter "roIndex", type "Integer"
tbricks::ParameterDefinition RoIndex();

// Strategy parameter "EODRealized", type "Double"
tbricks::ParameterDefinition EODRealized();

// Strategy parameter "VWAPWindowSize (seconds)", type "Integer"
tbricks::ParameterDefinition VWAPWindowSize_LEFT_PARENTHESIS_Seconds_RIGHT_PARENTHESIS();

// Strategy parameter "ExchangeCost", type "Double"
tbricks::ParameterDefinition ExchangeCost();

// Strategy parameter "IsLP", type "Boolean"
tbricks::ParameterDefinition IsLP();

// Strategy parameter "QuoteSummary", type "Table"
tbricks::ParameterDefinition QuoteSummary();

// Strategy parameter "IsDS", type "Boolean"
tbricks::ParameterDefinition IsDS();

// Strategy parameter "IndicatorValue_3", type "Double"
tbricks::ParameterDefinition IndicatorValue3();

// Strategy parameter "ApplyExpandPosCriteria", type "Boolean"
tbricks::ParameterDefinition ApplyExpandPosCriteria();

// Strategy parameter "roLastPrice", type "Price"
tbricks::ParameterDefinition RoLastPrice();

// Strategy parameter "Alpha_1", type "Double"
tbricks::ParameterDefinition Alpha1();

// Strategy parameter "AskStatus", type "Integer"
tbricks::ParameterDefinition AskStatus();

// Strategy parameter "Turnover2", type "Double"
tbricks::ParameterDefinition Turnover2();

// Strategy parameter "WideBPs", type "Integer"
tbricks::ParameterDefinition WideBPs();

// Strategy parameter "FeeMICs", type "Integer"
tbricks::ParameterDefinition FeeMICs();

// Strategy parameter "Replicate hedge ivid", type "Boolean"
tbricks::ParameterDefinition ReplicateHedgeIvid();

// Strategy parameter "SGVWAP", type "Price"
tbricks::ParameterDefinition SGVWAP();

// Strategy parameter "IndicatorValue_1", type "Double"
tbricks::ParameterDefinition IndicatorValue1();

// Strategy parameter "EODPosition", type "Volume"
tbricks::ParameterDefinition EODPosition();

// Strategy parameter "SignalGeneratorParam", type "Integer"
tbricks::ParameterDefinition SignalGeneratorParam();

// Strategy parameter "roAvgVolumePerTrade", type "Volume"
tbricks::ParameterDefinition RoAvgVolumePerTrade();

// Strategy parameter "Enable Quote Ask Children", type "Boolean"
tbricks::ParameterDefinition EnableQuoteAskChildren();

// Strategy parameter "StreamDataMember", type "String"
tbricks::ParameterDefinition StreamDataMember();

// Strategy parameter "MIC_3", type "MIC"
tbricks::ParameterDefinition MIC_3();

// Strategy parameter "ClosePosInViewVolRatio", type "Double"
tbricks::ParameterDefinition ClosePosInViewVolRatio();

// Strategy parameter "EnableTrading_1", type "Boolean"
tbricks::ParameterDefinition EnableTrading1();

// Strategy parameter "HedgeExtraMarginRatio", type "Double"
tbricks::ParameterDefinition HedgeExtraMarginRatio();

// Strategy parameter "DisableSwx", type "Boolean"
tbricks::ParameterDefinition DisableSwx();

// Strategy parameter "FromDate", type "DateTime"
tbricks::ParameterDefinition FromDate();

// Strategy parameter "TestPluginDemoParam1", type "Double"
tbricks::ParameterDefinition TestPluginDemoParam1();

// Strategy parameter "Theo Method", type "Integer"
tbricks::ParameterDefinition TheoMethod();

// Strategy parameter "User Account_3", type "String"
tbricks::ParameterDefinition UserAccount3();

// Strategy parameter "DisableOmx", type "Boolean"
tbricks::ParameterDefinition DisableOmx();

// Strategy parameter "SpreadPrice", type "Double"
tbricks::ParameterDefinition SpreadPrice();

// Strategy parameter "roBestBidVolume", type "Volume"
tbricks::ParameterDefinition RoBestBidVolume();

// Strategy parameter "CashByCurrencyTable", type "Table"
tbricks::ParameterDefinition CashByCurrencyTable();

// Strategy parameter "TextOut", type "String"
tbricks::ParameterDefinition TextOut();

// Strategy parameter "StartTime", type "DateTime"
tbricks::ParameterDefinition StartTime();

}

namespace calculated_values {

// Calculated value  "TEST CALCULATED VALUE", type "Double"
tbricks::InstrumentCalculatedValueDefinition TEST_CALCULATED_VALUE();

}

namespace extra_data {

// Extra data "OrderType", type "Integer"
tbricks::String OrderType();

}

namespace roles {

// Role "TbricksUsers", description ""
tbricks::Uuid TbricksUsersIdentifier();

// Role "Sysadmins", description ""
tbricks::Uuid SysadminsIdentifier();

// Role "Risk", description ""
tbricks::Uuid RiskIdentifier();

}

}

#endif // __TBRICKS_DEFINITIONS_H__
