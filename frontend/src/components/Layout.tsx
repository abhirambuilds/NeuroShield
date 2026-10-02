import { NavLink } from "react-router-dom";
import type { ReactNode } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  Cpu,
  Database,
  Gauge,
  Info,
  LayoutDashboard,
  Sparkles,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/dataset", label: "Dataset", icon: Database },
  { to: "/training", label: "Training", icon: Cpu },
  { to: "/models", label: "Models", icon: BarChart3 },
  { to: "/prediction", label: "Prediction", icon: Sparkles },
  { to: "/uncertainty", label: "Uncertainty", icon: Gauge },
  { to: "/evaluation", label: "Evaluation", icon: Activity },
  { to: "/about", label: "About", icon: Info },
];

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="mx-auto flex min-h-screen max-w-[1400px] flex-col px-6">
      <header className="flex items-center justify-between border-b border-line py-5">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Brain className="h-8 w-8 text-accent-blue" />
          </div>
          <div>
            <div className="text-lg font-bold tracking-widest text-white">
              NEUROSHIELD
            </div>
            <div className="text-xs text-muted">
              Cognitive Workload Analysis Platform
            </div>
          </div>
        </div>
        <div className="hidden text-right sm:block">
          <div className="text-xs text-muted">
            Probabilistic Cognitive Workload &amp; Fatigue Analysis
          </div>
          <div className="mt-1 text-xs font-mono text-accent-purple">
            LDA · Bayesian · Linear SVM · Kernel SVM
          </div>
        </div>
      </header>

      <nav className="flex flex-wrap gap-1 border-b border-line py-3">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-accent-blue/15 text-white"
                  : "text-muted hover:text-white hover:bg-base-700/60"
              }`
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <main className="flex-1 py-8">{children}</main>

      <footer className="border-t border-line py-5 text-center text-xs text-muted">
        NeuroShield — research prototype. Metrics are computed from the loaded
        dataset; results are experimental and not diagnostic.
      </footer>
    </div>
  );
}
