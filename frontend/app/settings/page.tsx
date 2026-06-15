'use client';

import { type ApiKeys, loadApiKeys, saveApiKeys } from '@/lib/settings';
import { useEffect, useState } from 'react';
import toast from 'react-hot-toast';

const FIELDS: Array<{ key: keyof ApiKeys; label: string; placeholder: string }> = [
  { key: 'openai', label: 'OpenAI API Key', placeholder: 'sk-...' },
  { key: 'anthropic', label: 'Anthropic API Key', placeholder: 'sk-ant-...' },
  { key: 'google', label: 'Google API Key', placeholder: 'AIza...' },
];

export default function SettingsPage() {
  const [keys, setKeys] = useState<ApiKeys>({ openai: '', anthropic: '', google: '' });

  useEffect(() => {
    loadApiKeys()
      .then(setKeys)
      .catch(() => undefined);
  }, []);

  const handleSave = async () => {
    try {
      await saveApiKeys(keys);
      toast.success('Settings saved');
    } catch {
      toast.error('Could not save settings');
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-2">Settings</h1>
      <p className="text-muted-foreground mb-8">
        API keys are stored in your browser only and are sent to the backend solely to generate
        reports.
      </p>

      <div className="max-w-2xl space-y-6">
        <div className="bg-card p-6 rounded-lg border border-border">
          <h2 className="text-xl font-semibold mb-4">API Keys</h2>
          <div className="space-y-4">
            {FIELDS.map((field) => (
              <div key={field.key}>
                <label className="block text-sm font-medium mb-2" htmlFor={field.key}>
                  {field.label}
                </label>
                <input
                  id={field.key}
                  type="password"
                  placeholder={field.placeholder}
                  value={keys[field.key]}
                  onChange={(e) => setKeys((prev) => ({ ...prev, [field.key]: e.target.value }))}
                  className="w-full px-4 py-2 border border-border rounded-md focus:ring-2 focus:ring-ring focus:border-ring"
                />
              </div>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={() => void handleSave()}
          className="w-full py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}
