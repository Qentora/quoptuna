'use client';

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

  useEffect(() => {
    loadApiKeys()
      .then((loaded) => {
        setKeys(loaded);
        setSaved(loaded);
      })
      .catch(() => undefined);
  }, []);

  const dirty = FIELDS.some((f) => keys[f.key] !== saved[f.key]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveApiKeys(keys);
      setSaved(keys);
      toast.success('Settings saved');
    } catch {
      toast.error('Could not save settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Settings</h1>
      <p className="text-muted-foreground mb-8">
        API keys are stored encrypted in your browser only and are sent to the backend solely to
        generate reports.
      </p>

      <div className="space-y-6">
        <div className="bg-card p-6 rounded-lg border border-border">
          <h2 className="text-xl font-semibold mb-4">API Keys</h2>
          <div className="space-y-5">
            {FIELDS.map((field) => {
              const isSet = saved[field.key].trim().length > 0;
              const show = visible[field.key];
              return (
                <div key={field.key}>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium" htmlFor={field.key}>
                      {field.label}
                    </label>
                    <span
                      className={`inline-flex items-center gap-1.5 text-xs font-medium ${
                        isSet ? 'text-green-600' : 'text-muted-foreground'
                      }`}
                    >
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${
                          isSet ? 'bg-green-500' : 'bg-muted-foreground/40'
                        }`}
                      />
                      {isSet ? 'Configured' : 'Not set'}
                    </span>
                  </div>
                  <div className="relative">
                    <input
                      id={field.key}
                      type={show ? 'text' : 'password'}
                      placeholder={field.placeholder}
                      value={keys[field.key]}
                      onChange={(e) =>
                        setKeys((prev) => ({ ...prev, [field.key]: e.target.value }))
                      }
                      className="w-full px-4 py-2 pr-11 border border-border rounded-md bg-background focus:ring-2 focus:ring-ring focus:border-ring"
                    />
                    <button
                      type="button"
                      onClick={() =>
                        setVisible((prev) => ({ ...prev, [field.key]: !prev[field.key] }))
                      }
                      aria-label={show ? 'Hide key' : 'Show key'}
                      className="absolute inset-y-0 right-0 flex items-center px-3 text-muted-foreground hover:text-foreground"
                    >
                      {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1.5">{field.helper}</p>
                </div>
              );
            })}
          </div>
        </div>

        <button
          type="button"
          onClick={() => void handleSave()}
          disabled={!dirty || saving}
          className="w-full py-3 bg-primary text-primary-foreground rounded-md transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {saving ? 'Saving…' : dirty ? 'Save Settings' : 'Saved'}
        </button>
      </div>
    </div>
  );
}
