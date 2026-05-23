import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

// Load environment variables from .env if present
dotenv.config();

// Simple in-memory rate limiting map
// Maps IP to array of request timestamps
const rateLimitMap = new Map<string, number[]>();
const RATE_LIMIT_WINDOW_MS = 60000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 20; // 20 requests per minute

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  if (!rateLimitMap.has(ip)) {
    rateLimitMap.set(ip, [now]);
    return false;
  }
  const timestamps = rateLimitMap.get(ip)!;
  // Filter out timestamps older than the window
  const activeTimestamps = timestamps.filter(t => now - t < RATE_LIMIT_WINDOW_MS);
  if (activeTimestamps.length >= RATE_LIMIT_MAX_REQUESTS) {
    rateLimitMap.set(ip, activeTimestamps);
    return true;
  }
  activeTimestamps.push(now);
  rateLimitMap.set(ip, activeTimestamps);
  return false;
}

// Lazy initializer for Gemini client to prevent startup crashes if GEMINI_API_KEY is not defined yet
let geminiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey.trim() === '') {
    throw new Error('GEMINI_API_KEY is not configured. Please open Settings > Secrets and set your GEMINI_API_KEY.');
  }
  
  if (!geminiClient) {
    geminiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return geminiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware for parsing JSON content
  app.use(express.json({ limit: '5mb' }));

  // API Route: AI conversion endpoint
  app.post('/api/convert', async (req: express.Request, res: express.Response) => {
    const ip = req.ip || 'unknown';
    if (isRateLimited(ip)) {
      res.status(429).json({
        success: false,
        error: 'Too many requests. Please wait a minute and try again.',
      });
      return;
    }

    const { text, mode, tone, targetLanguage, customInstructions } = req.body;

    if (!text || typeof text !== 'string' || text.trim() === '') {
      res.status(400).json({ success: false, error: 'Input text is required.' });
      return;
    }

    if (text.length > 25000) {
      res.status(400).json({ success: false, error: 'Input text exceeds the maximum character limit of 25,000 characters.' });
      return;
    }

    try {
      // Lazy init and key check
      const ai = getGeminiClient();

      // Formulate detailed guide based on selected mode
      let modeInstructions = '';
      switch (mode) {
        case 'rewrite':
          modeInstructions = 'Rephrase and rewrite the text clearly, improving flow, word choice, and legibility while preserving the fundamental core meaning of the original message.';
          break;
        case 'email':
          modeInstructions = 'Rewrite and format the input into a polished, extremely professional email. Structure this with a clear and attention-grabbing Subject Line (clearly labeled at the top), a polite opening greeting, an elegantly articulated body with bullet points if helpful for legibility, and a professional closing sign-off.';
          break;
        case 'social':
          modeInstructions = 'Rewrite the content to make it a perfect, highly engaging social media post (optimizing for platforms like LinkedIn, X/Twitter, or Instagram). Use appropriate negative space, add a touch of friendly character with relevant emojis, and supply 3-5 highly relevant hashtags at the bottom.';
          break;
        case 'summarize':
          modeInstructions = 'Summarize the text completely. Provide a concise introductory paragraph capturing the main premise, followed by a neatly formatted, high-value bulleted list highlighting the key takeaways and actionable metrics of the text.';
          break;
        case 'expand':
          modeInstructions = 'Elaborate on the topic, add rich supporting details, contextual explanations, and articulate any underlying concepts. Ensure that you do not add redundant filler, fluff, or generic text, but actually deepen the quality and information density of the content.';
          break;
        case 'formal':
          modeInstructions = 'Refine the vocabulary and syntax to be significantly more formal. Eliminate casual speech, idioms, and colloquialisms. Structure sentences to reflect premium corporate, diplomatic, legal, or high-tier academic presentation.';
          break;
        case 'casual':
          modeInstructions = 'Transcribe the core message to sound highly approachable, warm, causal, and friendly. Adapt a conversational dynamic as if speaking directly to a valued peer or close teammate, while keeping the absolute clarity of the message intact.';
          break;
        case 'seo':
          modeInstructions = 'Optimize the content for modern Search Engine Optimization (SEO) standards. Structure with crisp subheadings (using markdown), brief scannable lists, clear formatting, and embed high-value contextual keyphrases. Finally, include a separate section at the bottom titled "🎯 Recommended SEO Target Keywords" referencing 4 key words or phrases used.';
          break;
        case 'grammar':
          modeInstructions = 'Meticulously review the text and fix all spelling errors, grammatical mistakes, awkward phrasing, misplaced commas, run-on sentences, and style flaws. Ensure that you retain the precise semantic context and sentence intents of the original author, without changing the structure more than necessary.';
          break;
        case 'creative':
          modeInstructions = 'Unshackle the content into standard creative writing excellence. Enrich the text with vivid descriptions, engaging narrative syntax, evocative imagery, and compelling metaphors to turn the core message into a highly artistic and memorable read.';
          break;
        default:
          modeInstructions = 'Rewrite and polish the text to look highly professional and pristine recursively.';
      }

      // Dynamic system instruction formulation
      let systemPrompt = `You are "ConvertPro-AI", a premium text engineering and optimization system. 
Your goal is to transform, rewrite, refine, or format text to match the user's selected objectives in a masterful, production-ready, publication-grade output.

Strict Instructions:
1. Adhere meticulously to the tone requested (if any).
2. If a target language is requested, write the final output strictly in that language.
3. If custom instructions are provided, prioritize them as a primary operational modifier.
4. Output ONLY the beautifully constructed result. Do not output conversational preamble, conversational postscript, meta-commentary, or introductory remarks like "Sure, here is your text rewrite:" or "Here is the summary of your email:". Start directly with the converted code/text.
5. Use clean, elegant Markdown formatting for headings, bold terms, lists, and quote blocks when suitable, to make it look incredibly readable.`;

      let promptModifier = `### Conversion Task:
Selected Mode: ${mode || 'rewrite'}
Goal: ${modeInstructions}
${tone ? `Desired Tone: ${tone}` : ''}
${targetLanguage && targetLanguage !== 'Original Language' ? `Target Output Language: ${targetLanguage}` : ''}
${customInstructions ? `Custom Operational Modifiers (MANDATORY override instructions): ${customInstructions}` : ''}

### Input Text:
"""
${text}
"""`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: promptModifier,
        config: {
          systemInstruction: systemPrompt,
          temperature: 0.7,
        },
      });

      const transformedText = response.text || '';

      res.json({
        success: true,
        transformedText: transformedText.trim(),
        originalLength: text.length,
        transformedLength: transformedText.length,
      });

    } catch (error: any) {
      console.error('Error in text transformation API:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'An unexpected error occurred during the conversion process.',
      });
    }
  });

  // Serve static files in production or hook Vite development server middleware in development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
    console.log('Running server in DEVELOPMENT mode with Vite dev middleware.');
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: express.Request, res: express.Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
    console.log('Running server in PRODUCTION mode serving statically built files.');
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening at http://localhost:${PORT}`);
  });
}

startServer().catch((e) => {
  console.error('Fatal server boot failure:', e);
});
