import { Sidebar } from '@/components/Sidebar';
import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { cn } from '@/lib/utils';
import { Providers } from './providers';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });

const jetBrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'QuOptuna',
  description: 'Quantum-enhanced machine learning with automated hyperparameter optimization',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(jetBrainsMono.variable, 'font-sans', inter.variable)}
    >
      <body>
        <Providers>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div data-app-scroll-container className="flex-1 overflow-auto">
              {children}
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
