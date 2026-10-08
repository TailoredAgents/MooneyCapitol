from __future__ import annotations

from enum import IntEnum


PROTOCOL_PACKAGE_VERSION = "0.90.0.0"
PROTOCOL_TEMPLATE_VERSION = "5.55"
TEMPLATE_ID_FIELD_NUMBER = 154467


class Plant(IntEnum):
    TICKER = 1
    ORDER = 2
    HISTORY = 3
    PNL = 4
    REPOSITORY = 5


class Template(IntEnum):
    # Shared requests/responses and unsolicited control messages.
    LOGIN_REQUEST = 10
    LOGIN_RESPONSE = 11
    LOGOUT_REQUEST = 12
    LOGOUT_RESPONSE = 13
    REFERENCE_DATA_REQUEST = 14
    REFERENCE_DATA_RESPONSE = 15
    SYSTEM_INFO_REQUEST = 16
    SYSTEM_INFO_RESPONSE = 17
    HEARTBEAT_REQUEST = 18
    HEARTBEAT_RESPONSE = 19
    GATEWAY_INFO_REQUEST = 20
    GATEWAY_INFO_RESPONSE = 21
    FLOW_CONTROL_REQUEST = 22
    FLOW_CONTROL_RESPONSE = 23
    SESSION_CONFIG_REQUEST = 24
    SESSION_CONFIG_RESPONSE = 25
    REJECT = 75
    USER_ACCOUNT_UPDATE = 76
    FORCED_LOGOUT = 77

    # Ticker Plant reference-data subset. Continuous market-data requests are
    # intentionally omitted from the read-only capture allowlist.
    INSTRUMENT_BY_UNDERLYING_REQUEST = 102
    INSTRUMENT_BY_UNDERLYING_RESPONSE = 103
    INSTRUMENT_BY_UNDERLYING_KEYS = 104
    TICK_SIZE_TABLE_REQUEST = 107
    TICK_SIZE_TABLE_RESPONSE = 108
    SEARCH_SYMBOLS_REQUEST = 109
    SEARCH_SYMBOLS_RESPONSE = 110
    PRODUCT_CODES_REQUEST = 111
    PRODUCT_CODES_RESPONSE = 112
    FRONT_MONTH_REQUEST = 113
    FRONT_MONTH_RESPONSE = 114
    AUX_REFERENCE_REQUEST = 121
    AUX_REFERENCE_RESPONSE = 122
    FRONT_MONTH_UPDATE = 159

    # Order Plant discovery, observation, replay and server pushes.
    LOGIN_INFO_REQUEST = 300
    LOGIN_INFO_RESPONSE = 301
    ACCOUNT_LIST_REQUEST = 302
    ACCOUNT_LIST_RESPONSE = 303
    ACCOUNT_RMS_REQUEST = 304
    ACCOUNT_RMS_RESPONSE = 305
    PRODUCT_RMS_REQUEST = 306
    PRODUCT_RMS_RESPONSE = 307
    ORDER_UPDATES_REQUEST = 308
    ORDER_UPDATES_RESPONSE = 309
    TRADE_ROUTES_REQUEST = 310
    TRADE_ROUTES_RESPONSE = 311
    NEW_ORDER_REQUEST = 312
    NEW_ORDER_RESPONSE = 313
    MODIFY_ORDER_REQUEST = 314
    MODIFY_ORDER_RESPONSE = 315
    CANCEL_ORDER_REQUEST = 316
    CANCEL_ORDER_RESPONSE = 317
    ORDER_HISTORY_DATES_REQUEST = 318
    ORDER_HISTORY_DATES_RESPONSE = 319
    SHOW_ORDERS_REQUEST = 320
    SHOW_ORDERS_RESPONSE = 321
    SHOW_ORDER_HISTORY_REQUEST = 322
    SHOW_ORDER_HISTORY_RESPONSE = 323
    ORDER_HISTORY_SUMMARY_REQUEST = 324
    ORDER_HISTORY_SUMMARY_RESPONSE = 325
    ORDER_HISTORY_DETAIL_REQUEST = 326
    ORDER_HISTORY_DETAIL_RESPONSE = 327
    OCO_ORDER_REQUEST = 328
    OCO_ORDER_RESPONSE = 329
    BRACKET_ORDER_REQUEST = 330
    BRACKET_ORDER_RESPONSE = 331
    UPDATE_TARGET_REQUEST = 332
    UPDATE_TARGET_RESPONSE = 333
    UPDATE_STOP_REQUEST = 334
    UPDATE_STOP_RESPONSE = 335
    BRACKET_UPDATES_REQUEST = 336
    BRACKET_UPDATES_RESPONSE = 337
    SHOW_BRACKETS_REQUEST = 338
    SHOW_BRACKETS_RESPONSE = 339
    SHOW_BRACKET_STOPS_REQUEST = 340
    SHOW_BRACKET_STOPS_RESPONSE = 341
    EXCHANGE_PERMISSIONS_REQUEST = 342
    EXCHANGE_PERMISSIONS_RESPONSE = 343
    LINK_ORDERS_REQUEST = 344
    LINK_ORDERS_RESPONSE = 345
    CANCEL_ALL_REQUEST = 346
    CANCEL_ALL_RESPONSE = 347
    EASY_TO_BORROW_REQUEST = 348
    EASY_TO_BORROW_RESPONSE = 349
    TRADE_ROUTE = 350
    RITHMIC_ORDER_NOTIFICATION = 351
    EXCHANGE_ORDER_NOTIFICATION = 352
    BRACKET_UPDATE = 353
    ACCOUNT_RMS_UPDATE = 356
    USER_INFO_UPDATE = 357
    ACCOUNT_AND_USER_UPDATE = 358
    MODIFY_ORDER_REFERENCE_REQUEST = 3500
    MODIFY_ORDER_REFERENCE_RESPONSE = 3501
    EXIT_POSITION_REQUEST = 3504
    EXIT_POSITION_RESPONSE = 3505
    REPLAY_EXECUTIONS_REQUEST = 3506
    REPLAY_EXECUTIONS_RESPONSE = 3507
    RMS_UPDATES_REQUEST = 3508
    RMS_UPDATES_RESPONSE = 3509
    USER_INFO_REQUEST = 3510
    USER_INFO_RESPONSE = 3511
    FILL_HISTORY_REQUEST = 3512
    FILL_HISTORY_RESPONSE = 3513
    USER_ENTITLEMENTS_REQUEST = 3516
    USER_ENTITLEMENTS_RESPONSE = 3517
    PLAYBACK_ACCOUNT_USERS_REQUEST = 3518
    PLAYBACK_ACCOUNT_USERS_RESPONSE = 3519
    ACCOUNT_USER_HISTORY_REQUEST = 3520
    ACCOUNT_USER_HISTORY_RESPONSE = 3521

    # PnL Plant.
    PNL_UPDATES_REQUEST = 400
    PNL_UPDATES_RESPONSE = 401
    PNL_SNAPSHOT_REQUEST = 402
    PNL_SNAPSHOT_RESPONSE = 403
    INSTRUMENT_PNL_UPDATE = 450
    ACCOUNT_PNL_UPDATE = 451


# Trading/account mutations are listed explicitly for auditable tests, while
# the policy below remains default-deny for every unlisted template as well.
BROKER_MUTATION_TEMPLATE_IDS = frozenset(
    {
        Template.NEW_ORDER_REQUEST,
        Template.MODIFY_ORDER_REQUEST,
        Template.CANCEL_ORDER_REQUEST,
        Template.OCO_ORDER_REQUEST,
        Template.BRACKET_ORDER_REQUEST,
        Template.UPDATE_TARGET_REQUEST,
        Template.UPDATE_STOP_REQUEST,
        Template.LINK_ORDERS_REQUEST,
        Template.CANCEL_ALL_REQUEST,
        Template.MODIFY_ORDER_REFERENCE_REQUEST,
        Template.EXIT_POSITION_REQUEST,
    }
)


READ_ONLY_SHARED_REQUEST_IDS = frozenset(
    {
        Template.LOGIN_REQUEST,
        Template.LOGOUT_REQUEST,
        Template.REFERENCE_DATA_REQUEST,
        Template.SYSTEM_INFO_REQUEST,
        Template.HEARTBEAT_REQUEST,
        Template.GATEWAY_INFO_REQUEST,
        Template.FLOW_CONTROL_REQUEST,
        Template.SESSION_CONFIG_REQUEST,
    }
)

READ_ONLY_ORDER_REQUEST_IDS = frozenset(
    {
        Template.LOGIN_INFO_REQUEST,
        Template.ACCOUNT_LIST_REQUEST,
        Template.ACCOUNT_RMS_REQUEST,
        Template.PRODUCT_RMS_REQUEST,
        Template.ORDER_UPDATES_REQUEST,
        Template.TRADE_ROUTES_REQUEST,
        Template.ORDER_HISTORY_DATES_REQUEST,
        Template.SHOW_ORDERS_REQUEST,
        Template.SHOW_ORDER_HISTORY_REQUEST,
        Template.ORDER_HISTORY_SUMMARY_REQUEST,
        Template.ORDER_HISTORY_DETAIL_REQUEST,
        Template.BRACKET_UPDATES_REQUEST,
        Template.SHOW_BRACKETS_REQUEST,
        Template.SHOW_BRACKET_STOPS_REQUEST,
        Template.EXCHANGE_PERMISSIONS_REQUEST,
        Template.EASY_TO_BORROW_REQUEST,
        Template.REPLAY_EXECUTIONS_REQUEST,
        Template.RMS_UPDATES_REQUEST,
        Template.USER_INFO_REQUEST,
        Template.FILL_HISTORY_REQUEST,
        Template.USER_ENTITLEMENTS_REQUEST,
        Template.PLAYBACK_ACCOUNT_USERS_REQUEST,
        Template.ACCOUNT_USER_HISTORY_REQUEST,
    }
)

READ_ONLY_PNL_REQUEST_IDS = frozenset(
    {Template.PNL_UPDATES_REQUEST, Template.PNL_SNAPSHOT_REQUEST}
)

READ_ONLY_REFERENCE_REQUEST_IDS = frozenset(
    {
        Template.REFERENCE_DATA_REQUEST,
        Template.INSTRUMENT_BY_UNDERLYING_REQUEST,
        Template.TICK_SIZE_TABLE_REQUEST,
        Template.SEARCH_SYMBOLS_REQUEST,
        Template.PRODUCT_CODES_REQUEST,
        Template.FRONT_MONTH_REQUEST,
        Template.AUX_REFERENCE_REQUEST,
    }
)

OUTBOUND_READ_ONLY_TEMPLATE_IDS = frozenset(
    READ_ONLY_SHARED_REQUEST_IDS
    | READ_ONLY_ORDER_REQUEST_IDS
    | READ_ONLY_PNL_REQUEST_IDS
    | READ_ONLY_REFERENCE_REQUEST_IDS
)


INBOUND_BINDINGS: dict[int, tuple[str, str]] = {
    11: ("response_login_pb2", "ResponseLogin"),
    13: ("response_logout_pb2", "ResponseLogout"),
    15: ("response_reference_data_pb2", "ResponseReferenceData"),
    17: ("response_rithmic_system_info_pb2", "ResponseRithmicSystemInfo"),
    19: ("response_heartbeat_pb2", "ResponseHeartbeat"),
    21: ("response_rithmic_system_gateway_info_pb2", "ResponseRithmicSystemGatewayInfo"),
    23: ("response_flow_control_pb2", "ResponseFlowControl"),
    25: ("response_session_config_pb2", "ResponseSessionConfig"),
    75: ("reject_pb2", "Reject"),
    76: ("user_account_update_pb2", "UserAccountUpdate"),
    77: ("forced_logout_pb2", "ForcedLogout"),
    103: ("response_get_instrument_by_underlying_pb2", "ResponseGetInstrumentByUnderlying"),
    104: ("response_get_instrument_by_underlying_keys_pb2", "ResponseGetInstrumentByUnderlyingKeys"),
    108: ("response_give_tick_size_type_table_pb2", "ResponseGiveTickSizeTypeTable"),
    110: ("response_search_symbols_pb2", "ResponseSearchSymbols"),
    112: ("response_product_codes_pb2", "ResponseProductCodes"),
    114: ("response_front_month_contract_pb2", "ResponseFrontMonthContract"),
    122: ("response_auxilliary_reference_data_pb2", "ResponseAuxilliaryReferenceData"),
    159: ("front_month_contract_update_pb2", "FrontMonthContractUpdate"),
    301: ("response_login_info_pb2", "ResponseLoginInfo"),
    303: ("response_account_list_pb2", "ResponseAccountList"),
    305: ("response_account_rms_info_pb2", "ResponseAccountRmsInfo"),
    307: ("response_product_rms_info_pb2", "ResponseProductRmsInfo"),
    309: ("response_subscribe_for_order_updates_pb2", "ResponseSubscribeForOrderUpdates"),
    311: ("response_trade_routes_pb2", "ResponseTradeRoutes"),
    319: ("response_show_order_history_dates_pb2", "ResponseShowOrderHistoryDates"),
    321: ("response_show_orders_pb2", "ResponseShowOrders"),
    323: ("response_show_order_history_pb2", "ResponseShowOrderHistory"),
    325: ("response_show_order_history_summary_pb2", "ResponseShowOrderHistorySummary"),
    327: ("response_show_order_history_detail_pb2", "ResponseShowOrderHistoryDetail"),
    337: ("response_subscribe_to_bracket_updates_pb2", "ResponseSubscribeToBracketUpdates"),
    339: ("response_show_brackets_pb2", "ResponseShowBrackets"),
    341: ("response_show_bracket_stops_pb2", "ResponseShowBracketStops"),
    343: ("response_list_exchange_permissions_pb2", "ResponseListExchangePermissions"),
    349: ("response_easy_to_borrow_list_pb2", "ResponseEasyToBorrowList"),
    350: ("trade_route_pb2", "TradeRoute"),
    351: ("rithmic_order_notification_pb2", "RithmicOrderNotification"),
    352: ("exchange_order_notification_pb2", "ExchangeOrderNotification"),
    353: ("bracket_updates_pb2", "BracketUpdates"),
    356: ("account_rms_updates_pb2", "AccountRmsUpdates"),
    357: ("user_info_update_pb2", "UserInfoUpdate"),
    358: ("account_and_user_update_pb2", "AccountAndUserUpdate"),
    3507: ("response_replay_executions_pb2", "ResponseReplayExecutions"),
    3509: ("response_account_rms_updates_pb2", "ResponseAccountRmsUpdates"),
    3511: ("response_get_user_info_pb2", "ResponseGetUserInfo"),
    3513: ("response_show_fill_history_pb2", "ResponseShowFillHistory"),
    3517: ("response_user_entitlements_pb2", "ResponseUserEntitlements"),
    3519: ("response_playback_acc_users_pb2", "ResponsePlaybackAccUsers"),
    3521: ("response_show_acc_user_history_pb2", "ResponseShowAccUserHistory"),
    401: ("response_pnl_position_updates_pb2", "ResponsePnLPositionUpdates"),
    403: ("response_pnl_position_snapshot_pb2", "ResponsePnLPositionSnapshot"),
    450: ("instrument_pnl_position_update_pb2", "InstrumentPnLPositionUpdate"),
    451: ("account_pnl_position_update_pb2", "AccountPnLPositionUpdate"),
}


OUTBOUND_BINDINGS: dict[int, tuple[str, str]] = {
    10: ("request_login_pb2", "RequestLogin"),
    12: ("request_logout_pb2", "RequestLogout"),
    14: ("request_reference_data_pb2", "RequestReferenceData"),
    16: ("request_rithmic_system_info_pb2", "RequestRithmicSystemInfo"),
    18: ("request_heartbeat_pb2", "RequestHeartbeat"),
    20: ("request_rithmic_system_gateway_info_pb2", "RequestRithmicSystemGatewayInfo"),
    22: ("request_flow_control_pb2", "RequestFlowControl"),
    24: ("request_session_config_pb2", "RequestSessionConfig"),
    102: ("request_get_instrument_by_underlying_pb2", "RequestGetInstrumentByUnderlying"),
    107: ("request_give_tick_size_type_table_pb2", "RequestGiveTickSizeTypeTable"),
    109: ("request_search_symbols_pb2", "RequestSearchSymbols"),
    111: ("request_product_codes_pb2", "RequestProductCodes"),
    113: ("request_front_month_contract_pb2", "RequestFrontMonthContract"),
    121: ("request_auxilliary_reference_data_pb2", "RequestAuxilliaryReferenceData"),
    300: ("request_login_info_pb2", "RequestLoginInfo"),
    302: ("request_account_list_pb2", "RequestAccountList"),
    304: ("request_account_rms_info_pb2", "RequestAccountRmsInfo"),
    306: ("request_product_rms_info_pb2", "RequestProductRmsInfo"),
    308: ("request_subscribe_for_order_updates_pb2", "RequestSubscribeForOrderUpdates"),
    310: ("request_trade_routes_pb2", "RequestTradeRoutes"),
    318: ("request_show_order_history_dates_pb2", "RequestShowOrderHistoryDates"),
    320: ("request_show_orders_pb2", "RequestShowOrders"),
    322: ("request_show_order_history_pb2", "RequestShowOrderHistory"),
    324: ("request_show_order_history_summary_pb2", "RequestShowOrderHistorySummary"),
    326: ("request_show_order_history_detail_pb2", "RequestShowOrderHistoryDetail"),
    336: ("request_subscribe_to_bracket_updates_pb2", "RequestSubscribeToBracketUpdates"),
    338: ("request_show_brackets_pb2", "RequestShowBrackets"),
    340: ("request_show_bracket_stops_pb2", "RequestShowBracketStops"),
    342: ("request_list_exchange_permissions_pb2", "RequestListExchangePermissions"),
    348: ("request_easy_to_borrow_list_pb2", "RequestEasyToBorrowList"),
    3506: ("request_replay_executions_pb2", "RequestReplayExecutions"),
    3508: ("request_account_rms_updates_pb2", "RequestAccountRmsUpdates"),
    3510: ("request_get_user_info_pb2", "RequestGetUserInfo"),
    3512: ("request_show_fill_history_pb2", "RequestShowFillHistory"),
    3516: ("request_user_entitlements_pb2", "RequestUserEntitlements"),
    3518: ("request_playback_acc_users_pb2", "RequestPlaybackAccUsers"),
    3520: ("request_show_acc_user_history_pb2", "RequestShowAccUserHistory"),
    400: ("request_pnl_position_updates_pb2", "RequestPnLPositionUpdates"),
    402: ("request_pnl_position_snapshot_pb2", "RequestPnLPositionSnapshot"),
}


def template_name(template_id: int) -> str:
    try:
        return Template(template_id).name
    except ValueError:
        return f"UNKNOWN_{template_id}"
