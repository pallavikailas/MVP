"""
PDF Compliance Report Generator — powered by ReportLab
Produces a downloadable A4 audit report covering all FairLens pipeline stages.
"""

from io import BytesIO
from datetime import datetime
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Colour palette ────────────────────────────────────────────────────────
DARK  = colors.HexColor("#111827")
PANEL = colors.HexColor("#f8fafc")
LENS  = colors.HexColor("#4f46e5")
GREEN = colors.HexColor("#15803d")
AMBER = colors.HexColor("#b45309")
RED   = colors.HexColor("#b91c1c")
LIGHT = colors.HexColor("#111827")
MUTED = colors.HexColor("#475569")
WHITE = colors.HexColor("#ffffff")


def _score_color(score: int) -> colors.HexColor:
    if score >= 80:
        return GREEN
    if score >= 60:
        return AMBER
    return RED


def _status_color(status: str) -> colors.HexColor:
    return GREEN if status == "PASS" else RED if status == "FAIL" else AMBER


def _sev_color(sev: str) -> colors.HexColor:
    mapping = {"critical": RED, "high": RED, "medium": AMBER, "low": GREEN}
    return mapping.get((sev or "").lower(), MUTED)


def _make_styles():
    h1 = ParagraphStyle("h1", fontSize=20, textColor=LIGHT, spaceAfter=4,
                         fontName="Helvetica-Bold", alignment=TA_LEFT)
    h2 = ParagraphStyle("h2", fontSize=13, textColor=LENS, spaceAfter=6,
                         fontName="Helvetica-Bold", spaceBefore=14)
    h3 = ParagraphStyle("h3", fontSize=10, textColor=LIGHT, spaceAfter=4,
                         fontName="Helvetica-Bold", spaceBefore=8)
    body = ParagraphStyle("body", fontSize=9, textColor=LIGHT, spaceAfter=4,
                           fontName="Helvetica", leading=14)
    mono = ParagraphStyle("mono", fontSize=8, textColor=MUTED, spaceAfter=2,
                           fontName="Courier", leading=12)
    caption = ParagraphStyle("caption", fontSize=8, textColor=MUTED,
                              fontName="Helvetica", alignment=TA_CENTER)
    meta = ParagraphStyle("meta", fontSize=9, textColor=MUTED, fontName="Helvetica",
                           spaceAfter=12)
    footer = ParagraphStyle("footer", fontSize=7, textColor=MUTED, fontName="Helvetica",
                             alignment=TA_CENTER)
    return h1, h2, h3, body, mono, caption, meta, footer


_TBL_BASE = [
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
    ("TEXTCOLOR", (0, 1), (-1, -1), LIGHT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
]

_HDR_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), LENS),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
]


def generate_pdf_report(result: dict[str, Any]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    h1, h2, h3, body, mono, caption, meta, footer = _make_styles()

    summary = result.get("summary", {})
    fair_score = result.get("fair_score", {})
    compliance_tags = result.get("compliance_tags", [])
    slice_metrics = result.get("slice_metrics", [])
    gemini = result.get("gemini_analysis", {})
    constitution = result.get("constitution") or {}
    proxy_hunt = result.get("proxy_hunt") or {}
    redteam = result.get("redteam") or {}
    model_probe = result.get("model_probe") or {}
    dataset_probe = result.get("dataset_probe") or {}
    audit_id = result.get("audit_id", "—")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    story = []

    # ── Cover header ────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("FairLens™", ParagraphStyle("brand", fontSize=26, textColor=LENS,
                                               fontName="Helvetica-Bold")),
        Paragraph(f"Audit ID: {audit_id}<br/>{now}", caption),
    ]]
    header_tbl = Table(header_data, colWidths=["70%", "30%"])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (0, -1), 12),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width="100%", thickness=2, color=LENS, spaceAfter=10))
    story.append(Paragraph("AI Bias &amp; Fairness Compliance Report", h1))
    story.append(Paragraph(
        f"Model type: <b>{result.get('model_type', '—')}</b> &nbsp;|&nbsp; "
        f"Dataset: <b>{result.get('dataset_source', '—')}</b> &nbsp;|&nbsp; "
        f"Samples: <b>{summary.get('total_samples', '—')}</b> &nbsp;|&nbsp; "
        f"Hotspots: <b>{summary.get('hotspot_count', 0)}</b>",
        meta,
    ))

    # ── Phase pipeline indicator ─────────────────────────────────────────────
    stages_done = []
    if model_probe:
        stages_done.append("Phase 1: Model Probe")
    if dataset_probe or slice_metrics:
        stages_done.append("Phase 2: Dataset Analysis")
    if constitution or proxy_hunt or slice_metrics:
        stages_done.append("Phase 3: Cross-Analysis")
    if redteam:
        stages_done.append("Red-Team & Remediation")
    story.append(Paragraph(
        "Completed phases: " + " → ".join(stages_done) if stages_done else "Audit in progress",
        ParagraphStyle("stages", fontSize=8, textColor=GREEN, fontName="Helvetica", spaceAfter=12),
    ))

    # ── FairScore ───────────────────────────────────────────────────────────
    if fair_score:
        score = fair_score.get("score", 0)
        label = fair_score.get("label", "—")
        sc = _score_color(score)
        story.append(Paragraph("Overall Fairness Score", h2))
        score_data = [[
            Paragraph(f'<font size="36" color="{sc.hexval()}">{score}</font>', caption),
            Paragraph(
                f'<font size="14" color="{sc.hexval()}"><b>{label}</b></font><br/>'
                f'<font size="9" color="{MUTED.hexval()}">out of 100 &nbsp;·&nbsp; '
                f'overall bias score {summary.get("overall_bias_score", "—")}</font>',
                body,
            ),
        ]]
        score_tbl = Table(score_data, colWidths=["20%", "80%"])
        score_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), PANEL),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(score_tbl)
        story.append(Spacer(1, 10))

    if model_probe:
        probe_summary = model_probe.get("summary", {})
        diagnostics = probe_summary.get("prediction_diagnostics") or model_probe.get("prediction_diagnostics") or {}
        story.append(Paragraph("Phase 1: Embedded Reference Probe", h2))
        story.append(Paragraph(
            f"Reference rows: <b>{model_probe.get('reference_dataset_size', '—')}</b> &nbsp;|&nbsp; "
            f"Biases detected: <b>{probe_summary.get('bias_count', 0)}</b> &nbsp;|&nbsp; "
            f"Most biased attribute: <b>{probe_summary.get('most_biased_attribute', '—')}</b>",
            meta,
        ))
        if diagnostics.get("collapsed_output") or diagnostics.get("near_constant_output"):
            story.append(Paragraph(
                f"<b>Probe validity warning:</b> {diagnostics.get('reason', 'Model output collapsed on the reference probe.')}",
                ParagraphStyle("probe_warning", parent=body, textColor=RED),
            ))
        model_probe_biases = model_probe.get("model_biases") or []
        if model_probe_biases:
            rows = [["Attribute", "Type", "Magnitude", "Severity"]]
            for item in model_probe_biases[:10]:
                rows.append([
                    Paragraph(str(item.get("attribute", "—")), mono),
                    Paragraph(str(item.get("type", "—")), mono),
                    Paragraph(f'{float(item.get("magnitude", 0)):.3f}', mono),
                    Paragraph(str(item.get("severity", "—")).upper(), mono),
                ])
            tbl = Table(rows, colWidths=["28%", "28%", "18%", "26%"])
            tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
            story.append(tbl)
            story.append(Spacer(1, 10))

    # ── Regulatory Compliance ───────────────────────────────────────────────
    if compliance_tags:
        story.append(Paragraph("Regulatory Compliance", h2))
        comp_rows = [["Regulation", "Domain", "Status", "SPD", "DI"]]
        for tag in compliance_tags:
            sc = _status_color(tag["status"])
            comp_rows.append([
                Paragraph(f"<b>{tag['label']}</b>", mono),
                Paragraph(tag["domain"], mono),
                Paragraph(f'<font color="{sc.hexval()}"><b>{tag["status"]}</b></font>', mono),
                Paragraph(str(tag.get("worst_spd", "—")), mono),
                Paragraph(str(tag.get("worst_di", "—")), mono),
            ])
        comp_tbl = Table(comp_rows, colWidths=["30%", "22%", "13%", "17%", "18%"])
        comp_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
        story.append(comp_tbl)
        story.append(Spacer(1, 10))

    # ── Phase 1: Model Probe ─────────────────────────────────────────────────
    if model_probe:
        story.append(Paragraph("Phase 1: Model Probe (Embedded Reference Dataset)", h2))
        mp_summary = model_probe.get("summary", {})
        ref_size = model_probe.get("reference_dataset_size", "—")
        ref_cols = ", ".join(model_probe.get("reference_protected_cols") or []) or "—"
        story.append(Paragraph(
            f'Reference dataset: <b>{ref_size}</b> rows &nbsp;|&nbsp; '
            f'Protected cols: <b>{ref_cols}</b> &nbsp;|&nbsp; '
            f'Biases found: <b>{mp_summary.get("bias_count", 0)}</b>',
            meta,
        ))
        if model_probe.get("degenerate"):
            story.append(Paragraph(
                "<b>Model Probe Inconclusive — Synthetic Data Out-of-Distribution</b>", body
            ))
            story.append(Paragraph(str(model_probe.get("degenerate_message", "")), body))
        else:
            mp_biases = model_probe.get("model_biases", [])
            if mp_biases:
                story.append(Paragraph("Hidden Biases Detected by Model Probe", h3))
                bias_rows = [["Attribute", "Type", "Severity", "Magnitude", "Source"]]
                for b in mp_biases:
                    sc = _sev_color(b.get("severity", ""))
                    bias_rows.append([
                        Paragraph(str(b.get("attribute", "—")), mono),
                        Paragraph(str(b.get("type", "—")), mono),
                        Paragraph(f'<font color="{sc.hexval()}">{(b.get("severity") or "—").upper()}</font>', mono),
                        Paragraph(f'{b.get("magnitude", 0):.3f}', mono),
                        Paragraph(str(b.get("source", "—")), mono),
                    ])
                bias_tbl = Table(bias_rows, colWidths=["20%", "22%", "14%", "14%", "30%"])
                bias_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
                story.append(bias_tbl)
        story.append(Spacer(1, 10))

    # ── Phase 3: Cross-Analysis — Cartography Gemini Analysis ───────────────
    if gemini and gemini.get("headline"):
        story.append(Paragraph("Phase 3: Cross-Analysis — AI Bias Synthesis", h2))
        story.append(Paragraph(f'<b>{gemini.get("headline", "")}</b>', body))
        story.append(Paragraph(
            f'Severity: <b>{gemini.get("severity", "—")}</b> &nbsp;|&nbsp; '
            f'Bias type: <b>{gemini.get("bias_type", "—")}</b> &nbsp;|&nbsp; '
            f'Most affected: <b>{gemini.get("most_affected_group", "—")}</b>',
            mono,
        ))
        for f in gemini.get("key_findings", []):
            story.append(Paragraph(f"• {f}", body))
        if gemini.get("real_world_impact"):
            story.append(Paragraph(f'<b>Impact:</b> {gemini["real_world_impact"]}', body))
        if gemini.get("legal_risk"):
            story.append(Paragraph(f'<b>Legal risk:</b> {gemini["legal_risk"]}', body))
        if gemini.get("recommended_action"):
            story.append(Paragraph(f'<b>Recommended action:</b> {gemini["recommended_action"]}', body))
        story.append(Spacer(1, 6))

    # ── Phase 3: Top Bias Findings ────────────────────────────────────────────
    if slice_metrics:
        story.append(Paragraph("Phase 3: Top Bias Findings by Magnitude (Cross-Analysis)", h2))
        metric_rows = [["Demographic Slice", "SPD", "DI", "Pos. Rate", "Samples", "Flagged"]]
        for m in slice_metrics[:15]:
            flagged = "YES" if m.get("flagged") else "OK"
            fc = RED if m.get("flagged") else GREEN
            di_val = m.get("disparate_impact")
            di_str = f'{di_val:.4f}' if di_val is not None else "N/A"
            metric_rows.append([
                Paragraph(m["label"], mono),
                Paragraph(f'{m["statistical_parity_diff"]:+.4f}', mono),
                Paragraph(di_str, mono),
                Paragraph(f'{m["positive_rate"]:.2%}', mono),
                Paragraph(str(m["size"]), mono),
                Paragraph(f'<font color="{fc.hexval()}">{flagged}</font>', mono),
            ])
        metric_tbl = Table(metric_rows, colWidths=["34%", "12%", "11%", "14%", "12%", "17%"])
        metric_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
        story.append(metric_tbl)
        story.append(Spacer(1, 10))

    # ── Phase 3: Counterfactual Constitution ────────────────────────────────
    if constitution:
        story.append(PageBreak())
        story.append(Paragraph("Phase 3: Counterfactual Constitution", h2))

        const_summary = constitution.get("summary", {})
        if const_summary:
            story.append(Paragraph(
                f'Total counterfactual pairs: <b>{const_summary.get("total_cf_pairs", "—")}</b> &nbsp;|&nbsp; '
                f'Decision flip rate: <b>{const_summary.get("flip_rate", 0):.1%}</b> &nbsp;|&nbsp; '
                f'Most sensitive attribute: <b>{const_summary.get("most_sensitive_attribute", "—")}</b>',
                meta,
            ))

        patterns = constitution.get("patterns", [])
        if patterns:
            story.append(Paragraph("Demographic Sensitivity Index", h3))
            pat_rows = [["Attribute", "Flip Rate", "Avg Prob Shift", "Severity", "Bias Direction"]]
            for p in patterns:
                sc = _sev_color(p.get("severity", ""))
                pat_rows.append([
                    Paragraph(p.get("attribute", ""), mono),
                    Paragraph(f'{p.get("flip_rate", 0):.1%}', mono),
                    Paragraph(f'±{p.get("avg_probability_shift", 0):.3f}', mono),
                    Paragraph(f'<font color="{sc.hexval()}">{p.get("severity", "—").upper()}</font>', mono),
                    Paragraph(str(p.get("bias_direction", "—")), mono),
                ])
            pat_tbl = Table(pat_rows, colWidths=["22%", "16%", "18%", "16%", "28%"])
            pat_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
            story.append(pat_tbl)
            story.append(Spacer(1, 8))

        # Constitution sections (Gemini-generated)
        for section in constitution.get("sections", []):
            title = section.get("title", "")
            content = section.get("content", "").strip()
            if not content:
                continue
            story.append(Paragraph(title, h3))
            # Render line-by-line (skip markdown table dividers)
            for line in content.split("\n"):
                if not line.strip() or line.strip().startswith("|---"):
                    continue
                if line.startswith("|"):
                    # Simple table row — render as plain text
                    story.append(Paragraph(line.replace("|", " | ").strip(), mono))
                elif line.startswith(">"):
                    story.append(Paragraph(line.lstrip("> "), body))
                else:
                    story.append(Paragraph(line, body))

    # ── Phase 3: Proxy Variable Hunt ────────────────────────────────────────
    if proxy_hunt:
        story.append(PageBreak())
        story.append(Paragraph("Phase 3: Proxy Variable Hunt", h2))

        proxy_summary = proxy_hunt.get("summary", {})
        if proxy_summary:
            story.append(Paragraph(
                f'Proxy variables found: <b>{proxy_summary.get("proxy_count", "—")}</b> &nbsp;|&nbsp; '
                f'Severity: <b>{proxy_summary.get("severity", "—")}</b>',
                meta,
            ))

        proxy_vars = proxy_hunt.get("proxy_variables", [])
        if proxy_vars:
            story.append(Paragraph("Identified Proxy Variables", h3))
            pv_rows = [["Variable", "Correlation", "Severity", "Protected Attr", "Mechanism"]]
            for pv in proxy_vars[:20]:
                sc = _sev_color(pv.get("severity", ""))
                pv_rows.append([
                    Paragraph(str(pv.get("variable", "—")), mono),
                    Paragraph(f'{pv.get("correlation", 0):.3f}' if pv.get("correlation") is not None else "—", mono),
                    Paragraph(f'<font color="{sc.hexval()}">{(pv.get("severity") or "—").upper()}</font>', mono),
                    Paragraph(str(pv.get("protected_attribute", "—")), mono),
                    Paragraph(str(pv.get("mechanism", "—"))[:60], mono),
                ])
            pv_tbl = Table(pv_rows, colWidths=["18%", "14%", "12%", "18%", "38%"])
            pv_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
            story.append(pv_tbl)
            story.append(Spacer(1, 8))

        # Gemini proxy analysis narrative
        proxy_gemini = proxy_hunt.get("gemini_analysis", {})
        if proxy_gemini and proxy_gemini.get("headline"):
            story.append(Paragraph(f'<b>{proxy_gemini["headline"]}</b>', body))
            for f in proxy_gemini.get("key_findings", []):
                story.append(Paragraph(f"• {f}", body))
            if proxy_gemini.get("recommended_action"):
                story.append(Paragraph(f'<b>Recommended action:</b> {proxy_gemini["recommended_action"]}', body))

    # ── Red-Team & Remediation ───────────────────────────────────────────────
    if redteam:
        story.append(PageBreak())
        story.append(Paragraph("Red-Team Adversarial Audit &amp; Remediation", h2))

        # The state can come in two shapes:
        # 1. safe_state directly (has "iteration", "validation_results", "patch_results")
        # 2. final_report dict (has "iterations", "validation", "patches_applied")
        rt_report = redteam.get("final_report") or redteam
        validation = rt_report.get("validation") or redteam.get("validation_results") or {}
        plan = rt_report.get("mitigation_plan") or redteam.get("mitigation_plan") or []
        patch_results = redteam.get("patch_results") or {}
        patches_applied = rt_report.get("patches_applied") or len(patch_results.get("applied", []))
        biases_improved = rt_report.get("biases_improved") or len(validation.get("improved", []))
        iterations = rt_report.get("iterations") or redteam.get("iteration") or "—"
        confirmed_count = rt_report.get("biases_targeted") or len(redteam.get("confirmed_biases") or [])
        log_summary = rt_report.get("log_summary") or redteam.get("log", [])[-10:]

        story.append(Paragraph(
            f'Iterations: <b>{iterations}</b> &nbsp;|&nbsp; '
            f'Biases targeted: <b>{confirmed_count}</b> &nbsp;|&nbsp; '
            f'Patches applied: <b>{patches_applied}</b> &nbsp;|&nbsp; '
            f'Biases improved: <b>{biases_improved}</b>',
            meta,
        ))

        fairness_delta = rt_report.get("remediated_fairness") or {}
        if fairness_delta.get("before_avg_spd") is not None:
            story.append(Paragraph(
                f"Average measured disparity improved from <b>{fairness_delta['before_avg_spd']:.3f}</b> "
                f"to <b>{fairness_delta['after_avg_spd']:.3f}</b> "
                f"(delta <b>{fairness_delta['improvement']:.3f}</b>).",
                body,
            ))

        # Mitigation plan
        if plan:
            story.append(Paragraph("Mitigation Plan", h3))
            plan_rows = [["Attribute", "Strategy", "Disparity", "Rationale"]]
            for m in plan:
                plan_rows.append([
                    Paragraph(str(m.get("attribute", "—")), mono),
                    Paragraph(str(m.get("strategy", "—")), mono),
                    Paragraph(f'{m.get("disparity", 0):.3f}', mono),
                    Paragraph(str(m.get("rationale", "—"))[:80], mono),
                ])
            plan_tbl = Table(plan_rows, colWidths=["16%", "22%", "12%", "50%"])
            plan_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
            story.append(plan_tbl)
            story.append(Spacer(1, 8))

        # Validation results
        improved = validation.get("improved", [])
        regressed = validation.get("regressed", [])
        unchanged = validation.get("unchanged", [])

        if improved or regressed or unchanged:
            story.append(Paragraph("Validation Results (Before vs After Correction)", h3))
            val_rows = [["Attribute", "Before SPD", "After SPD", "Result"]]
            for item in improved:
                if isinstance(item, dict):
                    val_rows.append([
                        Paragraph(str(item.get("attribute", "—")), mono),
                        Paragraph(f'{item.get("before", 0):.3f}', mono),
                        Paragraph(f'{item.get("after", 0):.3f}', mono),
                        Paragraph(f'<font color="{GREEN.hexval()}">IMPROVED</font>', mono),
                    ])
                else:
                    val_rows.append([Paragraph(str(item), mono), Paragraph("—", mono), Paragraph("0.000", mono),
                                     Paragraph(f'<font color="{GREEN.hexval()}">IMPROVED</font>', mono)])
            for item in regressed:
                if isinstance(item, dict):
                    val_rows.append([
                        Paragraph(str(item.get("attribute", "—")), mono),
                        Paragraph(f'{item.get("before", 0):.3f}', mono),
                        Paragraph(f'{item.get("after", 0):.3f}', mono),
                        Paragraph(f'<font color="{RED.hexval()}">REGRESSED</font>', mono),
                    ])
            for item in unchanged:
                val_rows.append([Paragraph(str(item), mono), Paragraph("—", mono), Paragraph("—", mono),
                                 Paragraph(f'<font color="{AMBER.hexval()}">UNCHANGED</font>', mono)])
            if len(val_rows) > 1:
                val_tbl = Table(val_rows, colWidths=["28%", "20%", "20%", "32%"])
                val_tbl.setStyle(TableStyle(_HDR_STYLE + _TBL_BASE))
                story.append(val_tbl)
                story.append(Spacer(1, 8))

        artifact = rt_report.get("patched_model_artifact") or {}
        if artifact:
            story.append(Paragraph("Using The Remediated Model", h3))
            story.append(Paragraph(artifact.get("message", "No remediation artifact details available."), body))
            if artifact.get("available"):
                story.append(Paragraph(
                    f"Artifact: <b>{artifact.get('filename', 'fairlens-remediated-model.pkl')}</b> "
                    f"({artifact.get('format', 'pickle')})",
                    mono,
                ))

        # Activity log
        if log_summary:
            story.append(Paragraph("Agent Activity Log (last entries)", h3))
            for line in log_summary:
                story.append(Paragraph(str(line), mono))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceBefore=16, spaceAfter=6))
    story.append(Paragraph(
        f"Generated by FairLens™ · Audit {audit_id} · {now} · Powered by Gemini",
        footer,
    ))

    doc.build(story)
    return buf.getvalue()
