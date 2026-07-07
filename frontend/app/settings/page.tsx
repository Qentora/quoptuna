'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { PageShell } from '@/components/ui/page-shell';
import { DEFAULT_DATABASE_NAME, loadAppSettings, saveAppSettings } from '@/lib/appSettings';
import { type ApiKeys, loadApiKeys, saveApiKeys } from '@/lib/settings';
import { Eye, EyeOff } from 'lucide-react';
import { useEffect, useState } from 'react';
import toast from 'react-hot-toast';

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
  }, []);

  const dirty =
    FIELDS.some((f) => keys[f.key] !== saved[f.key]) || databaseName !== savedDatabaseName;

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveApiKeys(keys);
      setSaved(keys);
      const name = databaseName.trim() || DEFAULT_DATABASE_NAME;
      saveAppSettings({ databaseName: name });
      setDatabaseName(name);
      setSavedDatabaseName(name);
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
