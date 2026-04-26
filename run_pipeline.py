"""
run_pipeline.py
----------------
End-to-end manual test runner: WS3 → WS4 → WS5.
Run from the project root:

    conda run -n base python run_pipeline.py --resume Resumes/your_resume.pdf

Flags:
  --resume     Path to resume PDF or DOCX (required)
  --visa       Visa status string (default: "F-1 OPT")
  --sponsor    Pass this flag if you need sponsorship (default: True)
  --roles      Comma-separated target roles (default: "Data Engineer,Analytics Engineer")
  --locations  Comma-separated preferred locations (default: "Remote")
  --top        How many results to show (default: 15)
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from pipeline.ws3_resume_parser import parse_resume
from pipeline.ws4_job_matcher import match
from pipeline.ws5_confidence_scorer import score
from models.candidate_profile import UserInputs


def main():
    parser = argparse.ArgumentParser(description="Run full job matching pipeline")
    parser.add_argument("--resume",    required=True,  help="Path to resume PDF or DOCX")
    parser.add_argument("--visa",      default="F-1 OPT", help="Visa status")
    parser.add_argument("--no-sponsor", action="store_true", help="Does NOT need sponsorship")
    parser.add_argument("--roles",     default="Data Engineer,Analytics Engineer")
    parser.add_argument("--locations", default="Remote")
    parser.add_argument("--top",       type=int, default=15)
    args = parser.parse_args()

    inputs = UserInputs(
        visa_status=args.visa,
        requires_sponsorship=not args.no_sponsor,
        target_roles=[r.strip() for r in args.roles.split(",")],
        preferred_locations=[l.strip() for l in args.locations.split(",")],
        open_to_relocation=True,
    )

    log.info("=== Step 1/3: Parsing resume ===")
    profile = parse_resume(args.resume, inputs)
    log.info("Profile: %s | %s | %.1f yrs | skills: %s",
             profile.name, profile.seniority_level,
             profile.years_of_experience or 0,
             ", ".join(profile.technical_skills[:8]))

    log.info("=== Step 2/3: Matching jobs ===")
    matches = match(profile, top_n=50)

    log.info("=== Step 3/3: Scoring results ===")
    scored = score(matches, profile)

    # ── Print results ─────────────────────────────────────────────────────────
    sponsor_label = "INTERNATIONAL (with sponsorship)" if inputs.requires_sponsorship else "DOMESTIC (no sponsorship needed)"
    print(f"\n{'='*70}")
    print(f"  TOP {args.top} JOBS  —  {sponsor_label}")
    print(f"{'='*70}\n")

    for i, s in enumerate(scored[:args.top], 1):
        print(f"{i:2}. {s.title}")
        print(f"    Company  : {s.company_slug}")
        print(f"    Location : {s.location}")
        print(f"    Posted   : {s.posted_at}  ({s.days_since_posted}d ago)")
        print(f"    Score    : {s.final_score:.3f}  "
              f"[match={s.match_score:.3f}  recency={s.recency_score:.2f}"
              + (f"  visa={s.visa_confidence:.2f}  signal={s.sponsorship_signal}" if s.visa_confidence is not None else "")
              + "]")
        print(f"    Why      : {s.reasoning}")
        print(f"    Apply    : {s.apply_url}")
        print()

    print(f"{'='*70}")
    print(f"  Showing {min(args.top, len(scored))} of {len(scored)} matched jobs.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
