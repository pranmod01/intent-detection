#!/usr/bin/env python
"""
Master Pipeline Script

Run the complete motor intent detection analysis pipeline.

Usage:
    python run_pipeline.py --phase 1    # Run Phase 1 only
    python run_pipeline.py --phase 2    # Run Phase 2 only
    python run_pipeline.py --phase 3    # Run Phase 3 only
    python run_pipeline.py --phase 4    # Run Phase 4 (Temporal LSTM)
    python run_pipeline.py --all        # Run all phases
    python run_pipeline.py --demo       # Launch demo app
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_phase1():
    """Run Phase 1: Problem Demonstration."""
    print("\n" + "="*70)
    print("PHASE 1: Problem Demonstration")
    print("="*70)

    phase1_dir = Path(__file__).parent / 'phase1'

    # Step 1: Download and prepare PhysioNet data
    print("\n[1/3] Loading PhysioNet data and training baseline...")
    subprocess.run([sys.executable, str(phase1_dir / 'csp_lda_baseline.py')])

    # Step 2: Cross-dataset evaluation
    print("\n[2/3] Cross-dataset evaluation (the critical test)...")
    subprocess.run([sys.executable, str(phase1_dir / 'cross_dataset_evaluation.py')])

    # Step 3: Generate figures
    print("\n[3/3] Generating Phase 1 figures...")
    subprocess.run([sys.executable, str(phase1_dir / 'generate_figures.py')])


def run_phase2():
    """Run Phase 2: Multi-Condition Training."""
    print("\n" + "="*70)
    print("PHASE 2: Multi-Condition Training Solution")
    print("="*70)

    phase2_dir = Path(__file__).parent / 'phase2'

    # Step 1: CSP + LDA
    print("\n[1/2] Training multi-condition CSP+LDA...")
    subprocess.run([sys.executable, str(phase2_dir / 'multicond_csp_lda.py')])

    # Step 2: EEGNet
    print("\n[2/2] Training EEGNet classifier...")
    subprocess.run([sys.executable, str(phase2_dir / 'eegnet_classifier.py')])


def run_phase3():
    """Run Phase 3: Transfer Learning."""
    print("\n" + "="*70)
    print("PHASE 3: Transfer Learning Enhancement")
    print("="*70)

    phase3_dir = Path(__file__).parent / 'phase3'

    # Step 1: Autoencoder transfer learning
    print("\n[1/3] Transfer learning with autoencoder...")
    subprocess.run([sys.executable, str(phase3_dir / 'autoencoder_transfer.py')])

    # Step 2: Augmented training comparison
    print("\n[2/3] Augmented training comparison...")
    subprocess.run([sys.executable, str(phase3_dir / 'augmented_training.py')])

    # Step 3: Domain analysis
    print("\n[3/3] Domain analysis and visualization...")
    subprocess.run([sys.executable, str(phase3_dir / 'domain_analysis.py')])


def run_phase4():
    """Run Phase 4: Temporal Intent Detection."""
    print("\n" + "="*70)
    print("PHASE 4: Temporal Intent Detection (LSTM)")
    print("="*70)

    phase4_dir = Path(__file__).parent / 'phase4'

    print("\n[1/1] Training temporal LSTM with LOSO validation...")
    subprocess.run([sys.executable, str(phase4_dir / 'temporal_intent_lstm.py')])


def run_demo():
    """Run Interactive Demo."""
    print("\n" + "="*70)
    print("Interactive Demo")
    print("="*70)

    demo_dir = Path(__file__).parent / 'demo'

    print("\nLaunching Streamlit demo...")
    print("Open http://localhost:8501 in your browser")
    subprocess.run(['streamlit', 'run', str(demo_dir / 'demo_app.py')])


def run_existing_analysis():
    """Run existing EEG analysis pipeline."""
    print("\n" + "="*70)
    print("Running Existing Analysis Pipeline")
    print("="*70)

    analysis_dir = Path(__file__).parent / 'analysis'

    print("\n[1/2] Basic EEG analysis...")
    subprocess.run([sys.executable, str(analysis_dir / 'eeg_analysis.py')])

    print("\n[2/2] Enhanced analysis...")
    subprocess.run([sys.executable, str(analysis_dir / 'enhanced_analysis.py')])


def main():
    parser = argparse.ArgumentParser(
        description="Motor Intent Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py --phase 1     # Problem demonstration
    python run_pipeline.py --phase 2     # Multi-condition training
    python run_pipeline.py --phase 3     # Transfer learning
    python run_pipeline.py --phase 4     # Temporal LSTM
    python run_pipeline.py --all         # Run all phases
    python run_pipeline.py --demo        # Launch interactive demo
    python run_pipeline.py --existing    # Run existing analysis
        """
    )

    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                        help='Run specific phase (1-4)')
    parser.add_argument('--all', action='store_true',
                        help='Run all phases sequentially')
    parser.add_argument('--demo', action='store_true',
                        help='Launch interactive demo')
    parser.add_argument('--existing', action='store_true',
                        help='Run existing analysis pipeline')

    args = parser.parse_args()

    print("="*70)
    print("MOTOR INTENT DETECTION PIPELINE")
    print("Detecting Motor Intent in Hybrid Brain-Computer Interfaces")
    print("="*70)

    if args.existing:
        run_existing_analysis()
    elif args.demo:
        run_demo()
    elif args.all:
        run_phase1()
        run_phase2()
        run_phase3()
        run_phase4()
    elif args.phase:
        if args.phase == 1:
            run_phase1()
        elif args.phase == 2:
            run_phase2()
        elif args.phase == 3:
            run_phase3()
        elif args.phase == 4:
            run_phase4()
    else:
        parser.print_help()
        print("\n" + "-"*70)
        print("Quick Start:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run Phase 1: python run_pipeline.py --phase 1")
        print("  3. Run Phase 2: python run_pipeline.py --phase 2")
        print("  4. Run Phase 3: python run_pipeline.py --phase 3")
        print("  5. Run Phase 4: python run_pipeline.py --phase 4")
        print("  6. Launch demo: python run_pipeline.py --demo")
        print("-"*70)


if __name__ == "__main__":
    main()
