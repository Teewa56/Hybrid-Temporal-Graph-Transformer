import time
from synthetic_data_generator.config import CONFIG

from synthetic_data_generator.behavioral import (
    UserProfileGenerator, TransactionSequenceGenerator, AnomalyInjector
)
from synthetic_data_generator.graph import (
    GraphBuilder, MuleNetworkSimulator, FraudRingInjector
)
from synthetic_data_generator.payload import (
    LegitimatePayloadGenerator, PayloadAnomalyInjector
)
from synthetic_data_generator.sim_swap import (
    DeviceProfileGenerator, HandoverEventSimulator
)
from synthetic_data_generator.kyc import (
    DocumentMetadataGenerator, ForgerySimulator
)
from synthetic_data_generator.pipeline.export import export_all


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_all() -> dict:
    results = {}
    total_start = time.time()

    # ── 1. Behavioral ──────────────────────────────────────────
    _section("1/5  Behavioral Data (Transformer)")
    t0 = time.time()
    try:
        profile_gen = UserProfileGenerator()
        profiles = profile_gen.generate_batch()

        seq_gen = TransactionSequenceGenerator()
        sequences_by_user = [(seq_gen.generate_for_user(p), p) for p in profiles]

        injector = AnomalyInjector()
        sequences_with_labels = injector.inject_batch(sequences_by_user)

        all_transactions = [txn for seq in sequences_with_labels for txn in seq]
        fraud_count = sum(t["label"] for t in all_transactions)

        results["behavioral"] = {
            "users": len(profiles),
            "sequences": len(sequences_with_labels),
            "total_transactions": len(all_transactions),
            "fraud_transactions": fraud_count,
            "fraud_rate": round(fraud_count / max(len(all_transactions), 1), 4),
            "data": all_transactions,
            "profiles": profiles,
        }
        print(f"   {len(all_transactions):,} transactions | "
              f"Fraud rate: {results['behavioral']['fraud_rate']:.2%} | "
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   Behavioral failed: {e}")
        results["behavioral"] = {"error": str(e)}

    # ── 2. Graph ────────────────────────────────────────────────
    _section("2/5  Graph Data (GraphSAGE)")
    t0 = time.time()
    try:
        ring_injector = FraudRingInjector()
        graph_data = ring_injector.build_injected_graph()

        results["graph"] = {
            "n_nodes": graph_data["n_nodes"],
            "n_edges": graph_data["n_edges"],
            "fraud_rate": graph_data["fraud_rate"],
            "n_mule": graph_data["n_mule"],
            "data": graph_data,
        }
        print(f"   {graph_data['n_nodes']:,} nodes | "
              f"{graph_data['n_edges']:,} edges | "
              f"Fraud rate: {graph_data['fraud_rate']:.2%} | "
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   Graph failed: {e}")
        results["graph"] = {"error": str(e)}

    # ── 3. Payload ──────────────────────────────────────────────
    _section("3/5  Payload Data (CNN-GNN)")
    t0 = time.time()
    try:
        payload_gen = LegitimatePayloadGenerator()
        legit_payloads = payload_gen.generate_batch()

        payload_injector = PayloadAnomalyInjector()
        anomalous_payloads = payload_injector.inject_batch(legit_payloads)

        all_payloads = legit_payloads + anomalous_payloads
        fraud_count = sum(p.get("label", 0) for p in all_payloads)

        results["payload"] = {
            "total": len(all_payloads),
            "legit": len(legit_payloads),
            "anomalous": len(anomalous_payloads),
            "fraud_rate": round(fraud_count / max(len(all_payloads), 1), 4),
            "data": all_payloads,
        }
        print(f"   {len(all_payloads):,} payloads | "
              f"Fraud rate: {results['payload']['fraud_rate']:.2%} | "
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   Payload failed: {e}")
        results["payload"] = {"error": str(e)}

    # ── 4. SIM Swap ─────────────────────────────────────────────
    _section("4/5  SIM Swap Data (TSSGC)")
    t0 = time.time()
    try:
        device_gen = DeviceProfileGenerator()
        histories = device_gen.generate_batch()

        handover_sim = HandoverEventSimulator()
        histories_with_swaps = handover_sim.simulate_batch(histories)

        swap_count = sum(1 for h in histories_with_swaps if h.has_sim_swap)
        total_events = sum(len(h.events) for h in histories_with_swaps)

        results["sim_swap"] = {
            "users": len(histories_with_swaps),
            "sim_swap_events": swap_count,
            "total_device_events": total_events,
            "swap_rate": round(swap_count / max(len(histories_with_swaps), 1), 4),
            "data": histories_with_swaps,
        }
        print(f"   {len(histories_with_swaps):,} users | "
              f"{swap_count} swaps | "
              f"Swap rate: {results['sim_swap']['swap_rate']:.2%} | "
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   SIM Swap failed: {e}")
        results["sim_swap"] = {"error": str(e)}

    # ── 5. KYC ──────────────────────────────────────────────────
    _section("5/5  KYC Document Data (GAN + Autoencoder)")
    t0 = time.time()
    try:
        doc_gen = DocumentMetadataGenerator()
        legit_docs = doc_gen.generate_batch()

        forgery_sim = ForgerySimulator()
        forged_docs = forgery_sim.generate_forged_batch(legit_docs)

        all_docs = legit_docs + forged_docs
        fraud_count = sum(d.label for d in all_docs)

        results["kyc"] = {
            "total": len(all_docs),
            "legit": len(legit_docs),
            "forged": len(forged_docs),
            "fraud_rate": round(fraud_count / max(len(all_docs), 1), 4),
            "data": all_docs,
        }
        print(f"   {len(all_docs):,} documents | "
              f"Fraud rate: {results['kyc']['fraud_rate']:.2%} | "
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   KYC failed: {e}")
        results["kyc"] = {"error": str(e)}

    # ── Export ──────────────────────────────────────────────────
    _section("Exporting all datasets")
    export_all(results)

    total_time = time.time() - total_start
    _section(f" Done in {total_time:.1f}s")

    return results


if __name__ == "__main__":
    run_all()