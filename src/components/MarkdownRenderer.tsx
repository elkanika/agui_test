// src/components/MarkdownRenderer.tsx
"use client";

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { DynamicRenderer } from './dynamic/DynamicRenderer';

interface MarkdownRendererProps {
    content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
    return (
        <div className="markdown-content [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-sm [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-base">
            <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                    // Customize code blocks
                    code({ node, inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        const lang = match ? match[1] : '';

                        // Check for json-ui language
                        if (!inline && lang === 'json-ui') {
                            try {
                                const contentStr = String(children).replace(/\n$/, '');
                                const jsonData = JSON.parse(contentStr);

                                if (jsonData && jsonData.type && jsonData.data) {
                                    return (
                                        <div className="my-4 not-prose is-json-ui">
                                            <DynamicRenderer type={jsonData.type} data={jsonData.data} />
                                        </div>
                                    );
                                }
                            } catch (e) {
                                console.error("Failed to parse json-ui block:", e);
                                // Fallback to regular rendering if parse fails
                            }
                        }

                        return (
                            <code className={className} {...props}>
                                {children}
                            </code>
                        );
                    },
                    pre({ children }) {
                        // Check if the child (result of code component) is our dynamic UI component
                        try {
                            const child = React.Children.only(children) as React.ReactElement;
                            if (child && child.props && child.props.className && child.props.className.includes('is-json-ui')) {
                                return <>{children}</>;
                            }
                        } catch (e) {
                            // React.Children.only fails if there are multiple children or no children
                            // ignore and render default pre
                        }

                        return (
                            <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto my-2">
                                {children}
                            </pre>
                        );
                    },
                    // Customize paragraphs
                    p({ children }) {
                        return <p className="mb-2">{children}</p>;
                    },
                    // Customize lists
                    ul({ children }) {
                        return <ul className="list-disc list-inside mb-2">{children}</ul>;
                    },
                    ol({ children }) {
                        return <ol className="list-decimal list-inside mb-2">{children}</ol>;
                    },
                    // Customize headings
                    h1({ children }) {
                        return <h1 className="text-2xl font-bold mb-2">{children}</h1>;
                    },
                    h2({ children }) {
                        return <h2 className="text-xl font-bold mb-2">{children}</h2>;
                    },
                    h3({ children }) {
                        return <h3 className="text-lg font-bold mb-2">{children}</h3>;
                    },
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
}
