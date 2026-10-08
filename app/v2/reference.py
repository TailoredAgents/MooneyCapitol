from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol

from app.v2.domain.models import ContractMapping, ContractSpecification, FuturesContract


NQ_SPECIFICATION = ContractSpecification(
    product_code="NQ",
    exchange="CME",
    point_value=Decimal("20"),
    tick_size=Decimal("0.25"),
    tick_value=Decimal("5"),
)

MNQ_SPECIFICATION = ContractSpecification(
    product_code="MNQ",
    exchange="CME",
    point_value=Decimal("2"),
    tick_size=Decimal("0.25"),
    tick_value=Decimal("0.50"),
)

ES_SPECIFICATION = ContractSpecification(
    product_code="ES",
    exchange="CME",
    point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    tick_value=Decimal("12.50"),
)


class ContractRepository(Protocol):
    def save(self, contract: FuturesContract) -> None: ...
    def get(self, contract_id: str) -> FuturesContract | None: ...
    def save_mapping(self, mapping: ContractMapping) -> None: ...
    def mapped_contract(self, source_contract_id: str, target_product: str) -> FuturesContract | None: ...


class FuturesContractRegistry:
    """Provider-neutral in-memory registry; DB persistence is represented by V2 tables."""

    def __init__(self) -> None:
        self._specifications = {
            "NQ": NQ_SPECIFICATION,
            "MNQ": MNQ_SPECIFICATION,
            "ES": ES_SPECIFICATION,
        }
        self._contracts: dict[str, FuturesContract] = {}
        self._mappings: dict[tuple[str, str], str] = {}

    def specification(self, product_code: str) -> ContractSpecification:
        try:
            return self._specifications[product_code.upper()]
        except KeyError as exc:
            raise KeyError(f"unknown futures product: {product_code}") from exc

    def register_contract(
        self,
        *,
        contract_id: str,
        product_code: str,
        expiration: date,
        provider_symbols: dict[str, str] | None = None,
        first_trade_date: date | None = None,
        last_trade_date: date | None = None,
    ) -> FuturesContract:
        spec = self.specification(product_code)
        contract = FuturesContract(
            contract_id=contract_id,
            product_code=product_code.upper(),
            exchange=spec.exchange,
            expiration=expiration,
            specification=spec,
            provider_symbols=provider_symbols or {},
            first_trade_date=first_trade_date,
            last_trade_date=last_trade_date,
        )
        existing = self._contracts.get(contract_id)
        if existing is not None and existing != contract:
            raise ValueError(f"contract identity already registered with different facts: {contract_id}")
        self._contracts[contract_id] = contract
        return contract

    def get(self, contract_id: str) -> FuturesContract:
        if contract_id.upper() in self._specifications:
            raise ValueError("bare product code cannot resolve an expiration-specific contract")
        try:
            return self._contracts[contract_id]
        except KeyError as exc:
            raise KeyError(f"unknown futures contract: {contract_id}") from exc

    def map_same_expiry(self, source_contract_id: str, target_contract_id: str) -> ContractMapping:
        source = self.get(source_contract_id)
        target = self.get(target_contract_id)
        if source.expiration != target.expiration:
            raise ValueError("NQ/MNQ mappings must preserve exact expiry")
        mapping = ContractMapping(
            source_contract_id=source.contract_id,
            target_contract_id=target.contract_id,
            source_product=source.product_code,
            target_product=target.product_code,
            expiration=source.expiration,
        )
        self._mappings[(source.contract_id, target.product_code)] = target.contract_id
        self._mappings[(target.contract_id, source.product_code)] = source.contract_id
        return mapping

    def mapped_contract(self, source_contract_id: str, target_product: str) -> FuturesContract:
        target_id = self._mappings.get((source_contract_id, target_product.upper()))
        if target_id is None:
            raise KeyError(f"no same-expiry mapping for {source_contract_id} -> {target_product}")
        return self.get(target_id)
