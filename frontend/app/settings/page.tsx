'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { PageShell } from '@/components/ui/page-shell';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { type ReportPrompts, getReportPrompts } from '@/lib/api';
import {
  DEFAULT_CONVERGENCE_INTERVAL,
  DEFAULT_DATABASE_NAME,
  DEFAULT_DEV_TYPE,
  DEFAULT_MAX_STEPS,
  DEFAULT_MAX_VMAP,
  DEV_TYPE_OPTIONS,
  type DevType,
  MAX_VMAP_OPTIONS,
  loadAppSettings,
  saveAppSettings,
} from '@/lib/appSettings';
import {
  DEFAULT_MAX_TRIAL_ROWS,
  DEFAULT_REPORT_SETTINGS,
  PROMPT_DOCS,
  REPORT_SETTING_DOCS,
  REPORT_TOGGLES,
  type ReportSettings,
  type ReportToggle,
  loadReportSettings,
  saveReportSettings,
} from '@/lib/reportSettings';
import { type ApiKeys, loadApiKeys, saveApiKeys } from '@/lib/settings';
import { Eye, EyeOff, Minus, Plus, RotateCcw } from 'lucide-react';
import { useEffect, useState } from 'react';
import toast from 'react-hot-toast';

/** Integer input with -/+ stepper buttons; value is kept as a string state. */
function IntegerStepper({
  id,
  value,
  onChange,
  step,
  min = 1,
  placeholder,
}: {
  id: string;
  value: string;
  onChange: (next: string) => void;
  step: number;
  min?: number;
  placeholder?: string;
}) {
  const nudge = (direction: 1 | -1) => {
    const current = Math.floor(Number(value));
    const base = Number.isFinite(current) && current >= min ? current : min;
    onChange(String(Math.max(min, base + direction * step)));
  };
  return (
    <div className="flex w-full max-w-xs items-center gap-2">
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label="Decrease"
        onClick={() => nudge(-1)}
      >
        <Minus className="h-4 w-4" />
      </Button>
      <Input
        id={id}
        type="number"
        inputMode="numeric"
        min={min}
        step={step}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-center [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label="Increase"
        onClick={() => nudge(1)}
      >
        <Plus className="h-4 w-4" />
      </Button>
    </div>
  );
}

const FIELDS: Array<{ key: keyof ApiKeys; label: string; placeholder: string; helper: string }> = [
  {
    key: 'openai',
    label: 'OpenAI API Key',
    placeholder: 'sk-...',
    helper: 'Used for GPT models when generating reports.',
  },
  {
    key: 'anthropic',
    label: 'Anthropic API Key',
    placeholder: 'sk-ant-...',
    helper: 'Used for Claude models when generating reports.',
  },
  {
    key: 'google',
    label: 'Google API Key',
    placeholder: 'AIza...',
    helper: 'Used for Gemini models when generating reports.',
  },
];

const EMPTY: ApiKeys = { openai: '', anthropic: '', google: '' };

/** Local setting key -> the key the backend documents it under. */
const TOGGLE_API_KEYS: Record<ReportToggle | 'maxTrialRows', string> = {
  enableReview: 'enable_review',
  includeFairness: 'include_fairness',
  includeFairnessSearch: 'include_fairness_search',
  includePareto: 'include_pareto',
  includeShapDetail: 'include_shap_detail',
  includeTrialHistory: 'include_trial_history',
  includeStudyPlots: 'include_study_plots',
  attachFigures: 'attach_figures',
  maxTrialRows: 'max_trial_rows',
};

/**
 * One report toggle with its "what it does" / "when to use it" guidance.
 *
 * The clauses come from the backend's own settings metadata when it is
 * reachable, so the text a user reads is the text the agent is held to.
 */
function ToggleField({
  id,
  checked,
  onChange,
  label,
  description,
  when,
}: {
  id: string;
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  description: string;
  when: string;
}) {
  return (
    <Field>
      <div className="flex w-full items-start justify-between gap-4">
        <FieldLabel htmlFor={id} className="leading-snug">
          {label}
        </FieldLabel>
        <Switch id={id} checked={checked} onCheckedChange={onChange} />
      </div>
      <FieldDescription>{description}</FieldDescription>
      <FieldDescription className="text-muted-foreground/80">
        <span className="font-medium text-foreground/70">When to use: </span>
        {when}
      </FieldDescription>
    </Field>
  );
}

export default function SettingsPage() {
  const [keys, setKeys] = useState<ApiKeys>(EMPTY);
  const [saved, setSaved] = useState<ApiKeys>(EMPTY);
  const [visible, setVisible] = useState<Record<keyof ApiKeys, boolean>>({
    openai: false,
    anthropic: false,
    google: false,
  });
  const [saving, setSaving] = useState(false);
  const [databaseName, setDatabaseName] = useState(DEFAULT_DATABASE_NAME);
  const [savedDatabaseName, setSavedDatabaseName] = useState(DEFAULT_DATABASE_NAME);
  // Optimizer performance knobs; numeric fields kept as strings while editing.
  const [maxVmap, setMaxVmap] = useState(String(DEFAULT_MAX_VMAP));
  const [maxSteps, setMaxSteps] = useState(String(DEFAULT_MAX_STEPS));
  const [convergenceInterval, setConvergenceInterval] = useState(
    String(DEFAULT_CONVERGENCE_INTERVAL)
  );
  const [devType, setDevType] = useState<DevType>(DEFAULT_DEV_TYPE);
  const [savedOptimizer, setSavedOptimizer] = useState({
    maxVmap: String(DEFAULT_MAX_VMAP),
    maxSteps: String(DEFAULT_MAX_STEPS),
    convergenceInterval: String(DEFAULT_CONVERGENCE_INTERVAL),
    devType: DEFAULT_DEV_TYPE as DevType,
  });
  // Report generation: evidence toggles plus the two agent prompts.
  const [report, setReport] = useState<ReportSettings>(DEFAULT_REPORT_SETTINGS);
  const [savedReport, setSavedReport] = useState<ReportSettings>(DEFAULT_REPORT_SETTINGS);
  const [maxTrialRows, setMaxTrialRows] = useState(String(DEFAULT_MAX_TRIAL_ROWS));
  // Backend-served defaults: shown as the textarea placeholder and loaded by
  // "Start from default" so a custom prompt is an edit, not a rewrite.
  const [defaultPrompts, setDefaultPrompts] = useState<ReportPrompts | null>(null);

  useEffect(() => {
    getReportPrompts()
      .then(setDefaultPrompts)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    loadApiKeys()
      .then((loaded) => {
        setKeys(loaded);
        setSaved(loaded);
      })
      .catch(() => undefined);
    const appSettings = loadAppSettings();
    setDatabaseName(appSettings.databaseName);
    setSavedDatabaseName(appSettings.databaseName);
    const optimizer = {
      maxVmap: String(appSettings.maxVmap),
      maxSteps: String(appSettings.maxSteps),
      convergenceInterval: String(appSettings.convergenceInterval),
      devType: DEV_TYPE_OPTIONS.includes(appSettings.devType)
        ? appSettings.devType
        : DEFAULT_DEV_TYPE,
    };
    setMaxVmap(optimizer.maxVmap);
    setMaxSteps(optimizer.maxSteps);
    setConvergenceInterval(optimizer.convergenceInterval);
    setDevType(optimizer.devType);
    setSavedOptimizer(optimizer);
    const reportSettings = loadReportSettings();
    setReport(reportSettings);
    setSavedReport(reportSettings);
    setMaxTrialRows(String(reportSettings.maxTrialRows));
  }, []);

  const reportDirty =
    REPORT_TOGGLES.some((key) => report[key] !== savedReport[key]) ||
    report.analystInstructions !== savedReport.analystInstructions ||
    report.reviewerInstructions !== savedReport.reviewerInstructions ||
    maxTrialRows !== String(savedReport.maxTrialRows);

  const dirty =
    FIELDS.some((f) => keys[f.key] !== saved[f.key]) ||
    databaseName !== savedDatabaseName ||
    maxVmap !== savedOptimizer.maxVmap ||
    maxSteps !== savedOptimizer.maxSteps ||
    convergenceInterval !== savedOptimizer.convergenceInterval ||
    devType !== savedOptimizer.devType ||
    reportDirty;

  /** Guidance for one toggle, preferring the backend's copy over the bundled one. */
  const docFor = (key: ReportToggle | 'maxTrialRows') => {
    const local = REPORT_SETTING_DOCS[key];
    const remote = defaultPrompts?.settings.find((setting) => setting.key === TOGGLE_API_KEYS[key]);
    return {
      label: remote?.label ?? local.label,
      description: remote?.description ?? local.description,
      when: remote?.when ?? local.when,
    };
  };

  const parsePositiveInt = (raw: string, fallback: number): number => {
    const n = Math.floor(Number(raw));
    return Number.isFinite(n) && n >= 1 ? n : fallback;
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveApiKeys(keys);
      setSaved(keys);
      const name = databaseName.trim() || DEFAULT_DATABASE_NAME;
      const optimizerValues = {
        maxVmap: parsePositiveInt(maxVmap, DEFAULT_MAX_VMAP),
        maxSteps: parsePositiveInt(maxSteps, DEFAULT_MAX_STEPS),
        convergenceInterval: parsePositiveInt(convergenceInterval, DEFAULT_CONVERGENCE_INTERVAL),
        devType,
      };
      saveAppSettings({ databaseName: name, ...optimizerValues });
      setDatabaseName(name);
      setSavedDatabaseName(name);
      const optimizer = {
        maxVmap: String(optimizerValues.maxVmap),
        maxSteps: String(optimizerValues.maxSteps),
        convergenceInterval: String(optimizerValues.convergenceInterval),
        devType,
      };
      setMaxVmap(optimizer.maxVmap);
      setMaxSteps(optimizer.maxSteps);
      setConvergenceInterval(optimizer.convergenceInterval);
      setDevType(optimizer.devType);
      setSavedOptimizer(optimizer);
      const reportValues: ReportSettings = {
        ...report,
        maxTrialRows: Math.max(
          0,
          Number.isFinite(Math.floor(Number(maxTrialRows)))
            ? Math.floor(Number(maxTrialRows))
            : DEFAULT_MAX_TRIAL_ROWS
        ),
      };
      saveReportSettings(reportValues);
      setReport(reportValues);
      setSavedReport(reportValues);
      setMaxTrialRows(String(reportValues.maxTrialRows));
      toast.success('Settings saved');
    } catch {
      toast.error('Could not save settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <PageShell title="Settings" contentClassName="mx-auto max-w-2xl">
      <p className="mb-6 text-sm text-muted-foreground">
        API keys are stored encrypted in your browser only and are sent to the backend solely to
        generate reports.
      </p>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>API Keys</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            {FIELDS.map((field) => {
              const isSet = saved[field.key].trim().length > 0;
              const show = visible[field.key];
              return (
                <Field key={field.key}>
                  <div className="flex w-full items-center justify-between">
                    <FieldLabel htmlFor={field.key}>{field.label}</FieldLabel>
                    <Badge variant={isSet ? 'emerald' : 'secondary'}>
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${isSet ? 'bg-accent-emerald-foreground' : 'bg-muted-foreground/50'}`}
                      />
                      {isSet ? 'Configured' : 'Not set'}
                    </Badge>
                  </div>
                  <div className="relative">
                    <Input
                      id={field.key}
                      type={show ? 'text' : 'password'}
                      placeholder={field.placeholder}
                      value={keys[field.key]}
                      onChange={(e) =>
                        setKeys((prev) => ({ ...prev, [field.key]: e.target.value }))
                      }
                      className="pr-11"
                    />
                    <button
                      type="button"
                      onClick={() =>
                        setVisible((prev) => ({ ...prev, [field.key]: !prev[field.key] }))
                      }
                      aria-label={show ? 'Hide key' : 'Show key'}
                      className="absolute inset-y-0 right-0 flex items-center px-3 text-muted-foreground hover:text-foreground"
                    >
                      {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  <FieldDescription>{field.helper}</FieldDescription>
                </Field>
              );
            })}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Optimization Storage</CardTitle>
          </CardHeader>
          <CardContent>
            <Field>
              <FieldLabel htmlFor="database-name">Database Name</FieldLabel>
              <Input
                id="database-name"
                type="text"
                placeholder={DEFAULT_DATABASE_NAME}
                value={databaseName}
                onChange={(e) => setDatabaseName(e.target.value)}
              />
              <FieldDescription>
                Optuna SQLite database used for all optimization runs (stored server-side under{' '}
                <code className="font-mono">db/&lt;name&gt;.db</code>).
              </FieldDescription>
            </Field>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Optimizer Performance</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <Field>
              <FieldLabel htmlFor="max-vmap">Circuit vectorization (max_vmap)</FieldLabel>
              <Select value={maxVmap} onValueChange={setMaxVmap}>
                <SelectTrigger id="max-vmap" className="w-full max-w-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MAX_VMAP_OPTIONS.map((v) => (
                    <SelectItem key={v} value={String(v)}>
                      {v}
                      {v === DEFAULT_MAX_VMAP ? ' (default)' : v === 1 ? ' (slowest)' : ''}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FieldDescription>
                Quantum circuit evaluations vectorized per JAX call. Higher is much faster on small
                datasets but uses more memory. Must divide the batch size (32).
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="dev-type">Quantum simulator</FieldLabel>
              <Select value={devType} onValueChange={(v) => setDevType(v as DevType)}>
                <SelectTrigger id="dev-type" className="w-full max-w-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DEV_TYPE_OPTIONS.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                      {v === 'lightning.qubit' ? ' (faster)' : ' (default)'}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FieldDescription>
                PennyLane device used to simulate quantum circuits. lightning.qubit is a C++
                state-vector simulator that is usually faster; the backend falls back to
                default.qubit if it is unavailable.
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="max-steps">Max training steps per trial</FieldLabel>
              <IntegerStepper
                id="max-steps"
                value={maxSteps}
                onChange={setMaxSteps}
                step={500}
                placeholder={String(DEFAULT_MAX_STEPS)}
              />
              <FieldDescription>
                Caps how long each quantum model trains (model default is 10,000). Trials that hit
                the cap without converging are still scored.
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="convergence-interval">Convergence interval</FieldLabel>
              <IntegerStepper
                id="convergence-interval"
                value={convergenceInterval}
                onChange={setConvergenceInterval}
                step={25}
                placeholder={String(DEFAULT_CONVERGENCE_INTERVAL)}
              />
              <FieldDescription>
                Steps between flat-loss convergence checks and pruning reports. Lower values let
                converged trials exit sooner and give the pruner earlier decision points.
              </FieldDescription>
            </Field>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>AI Report Generation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <p className="text-sm text-muted-foreground">
              Reports are drafted by an analyst agent from a structured evidence bundle (every
              configuration option, the trial history, the Pareto front, metrics, SHAP and any
              fairness audit), then re-checked by a reviewer agent against the same bundle. These
              settings decide what enters that bundle and how the agents are instructed. Anything
              you exclude is recorded in the bundle as unavailable, so the agent says so instead of
              guessing.
            </p>

            {REPORT_TOGGLES.map((key) => {
              const doc = docFor(key);
              return (
                <ToggleField
                  key={key}
                  id={`report-${key}`}
                  checked={report[key]}
                  onChange={(next) => setReport((prev) => ({ ...prev, [key]: next }))}
                  label={doc.label}
                  description={doc.description}
                  when={doc.when}
                />
              );
            })}

            <Field>
              <FieldLabel htmlFor="report-max-trial-rows">
                {docFor('maxTrialRows').label}
              </FieldLabel>
              <IntegerStepper
                id="report-max-trial-rows"
                value={maxTrialRows}
                onChange={setMaxTrialRows}
                step={10}
                min={0}
                placeholder={String(DEFAULT_MAX_TRIAL_ROWS)}
              />
              <FieldDescription>{docFor('maxTrialRows').description}</FieldDescription>
              <FieldDescription className="text-muted-foreground/80">
                <span className="font-medium text-foreground/70">When to use: </span>
                {docFor('maxTrialRows').when}
              </FieldDescription>
            </Field>

            {(['analystInstructions', 'reviewerInstructions'] as const).map((key) => {
              const doc = PROMPT_DOCS[key];
              const fallback =
                key === 'analystInstructions'
                  ? defaultPrompts?.prompts.analyst
                  : defaultPrompts?.prompts.reviewer;
              const isCustom = report[key].trim().length > 0;
              return (
                <Field key={key}>
                  <div className="flex w-full items-center justify-between gap-2">
                    <FieldLabel htmlFor={`report-${key}`}>{doc.label}</FieldLabel>
                    <div className="flex items-center gap-2">
                      <Badge variant={isCustom ? 'emerald' : 'secondary'}>
                        {isCustom ? 'Custom' : 'Default'}
                      </Badge>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        disabled={!fallback && !isCustom}
                        onClick={() =>
                          setReport((prev) => ({
                            ...prev,
                            [key]: isCustom ? '' : (fallback ?? ''),
                          }))
                        }
                      >
                        <RotateCcw className="h-3.5 w-3.5" />
                        {isCustom ? 'Reset to default' : 'Start from default'}
                      </Button>
                    </div>
                  </div>
                  <Textarea
                    id={`report-${key}`}
                    value={report[key]}
                    onChange={(e) => setReport((prev) => ({ ...prev, [key]: e.target.value }))}
                    rows={isCustom ? 14 : 4}
                    className="font-mono text-xs"
                    placeholder={
                      fallback
                        ? `Using the built-in prompt:\n\n${fallback.slice(0, 400)}…`
                        : 'Leave empty to use the built-in prompt.'
                    }
                  />
                  <FieldDescription>{doc.description}</FieldDescription>
                  <FieldDescription className="text-muted-foreground/80">
                    <span className="font-medium text-foreground/70">When to use: </span>
                    {doc.when}
                  </FieldDescription>
                </Field>
              );
            })}

            {defaultPrompts && (
              <details className="rounded-md border border-border bg-muted/40 p-3">
                <summary className="cursor-pointer text-sm font-medium">
                  Markdown contract and report structure the agents must follow
                </summary>
                <p className="mt-2 text-xs text-muted-foreground">
                  Both prompts include these clauses, and the backend additionally repairs the
                  output (code fences, heading levels, table geometry, figure references) before the
                  report is saved — so a generated report always parses as GitHub-Flavored Markdown.
                </p>
                <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap font-mono text-xs text-muted-foreground">
                  {`${defaultPrompts.report_skeleton}\n${defaultPrompts.markdown_contract}`}
                </pre>
              </details>
            )}
          </CardContent>
        </Card>

        <Button
          type="button"
          onClick={() => void handleSave()}
          disabled={!dirty || saving}
          size="lg"
          className="w-full"
        >
          {saving ? 'Saving…' : dirty ? 'Save Settings' : 'Saved'}
        </Button>
      </div>
    </PageShell>
  );
}
