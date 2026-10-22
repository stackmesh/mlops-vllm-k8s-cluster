from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlops_cluster.catalog import load_models, with_concurrency
from mlops_cluster.pipeline import deploy_model
from mlops_cluster.planner import SlicePlan, plan_deployment
from mlops_cluster.registry import RegistryError
from mlops_cluster.render import write_manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlops-cluster")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print a GPU slice plan")
    _add_model_args(plan)
    plan.add_argument("--json", action="store_true", dest="as_json")

    render = sub.add_parser("render", help="write Kubernetes manifests for an accepted plan")
    _add_model_args(render)
    render.add_argument("--out", type=Path, required=True)

    deploy = sub.add_parser("deploy", help="plan, register the run, and render manifests")
    _add_model_args(deploy)
    deploy.add_argument("--out", type=Path)
    deploy.add_argument("--tracking-uri", default="")
    deploy.add_argument("--apply", action="store_true", help="kubectl apply the rendered manifests")

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    sub.add_parser("models", help="list catalog models")

    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            for name in sorted(load_models()):
                print(name)
            return 0
        if args.command == "serve":
            import uvicorn

            uvicorn.run("mlops_cluster.gateway:app", host=args.host, port=args.port)
            return 0
        if args.command == "plan":
            result = _plan(args)
            if args.as_json:
                print(json.dumps(result.as_dict(), indent=2))
            else:
                print(_format(result))
            return 0 if result.accepted else 1
        if args.command == "render":
            result = _plan(args)
            print(_format(result))
            if not result.accepted:
                return 1
            written = write_manifests(result, args.out)
            for name in sorted(written):
                print(args.out / name)
            return 0
        report = deploy_model(
            args.model,
            args.quant,
            args.replicas,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            out_dir=args.out,
            tracking_uri=args.tracking_uri or None,
            apply=args.apply,
        )
        print(_format_dict(report["plan"]))
        print(json.dumps({"registry": report["registry"], "manifests": report["manifests"]}, indent=2))
        return 0 if report["accepted"] else 1
    except KeyError:
        print(f"unknown model {args.model!r}", file=sys.stderr)
        return 2
    except (ValueError, RegistryError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", default="awq")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)


def _plan(args) -> SlicePlan:
    spec = with_concurrency(load_models()[args.model], args.max_model_len, args.max_num_seqs)
    from mlops_cluster.catalog import load_cluster

    return plan_deployment(spec, args.quant, args.replicas, load_cluster())


def _format(plan: SlicePlan) -> str:
    return _format_dict(plan.as_dict())


def _format_dict(plan: dict) -> str:
    estimate = plan.get("estimate") or {}
    lines = [
        f"model            {plan['model']}",
        f"quant            {plan['quant']}",
        f"accepted         {'yes' if plan['accepted'] else 'no'}",
        f"strategy         {plan.get('strategy') or '-'} {plan.get('mig_profile') or ''}".rstrip(),
        f"parallelism      tp={plan.get('tensor_parallel')} pp={plan.get('pipeline_parallel')}",
        f"replicas         {plan.get('replicas')}",
        f"physical gpus    {plan.get('physical_gpus')}",
        f"reserved vram    {plan.get('reserved_vram_gib')} GiB",
        (
            "estimate         "
            f"weights {estimate.get('weight_with_overhead_gib')} GiB + "
            f"kv {estimate.get('kv_gib')} GiB = {estimate.get('total_gib')} GiB"
        ),
        f"reason           {plan.get('reason')}",
    ]
    if plan.get("hint"):
        lines.append(f"hint             {plan['hint']}")
    if plan.get("alternatives"):
        lines.append("alternatives     " + "; ".join(plan["alternatives"]))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
