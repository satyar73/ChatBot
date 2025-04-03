# Format Templates Guide

This document explains how to use format templates for generating Google Slides and Google Docs content.

## Slides Format Templates

Slides format templates specify how content should be structured on slides. Each slide has a title and body.

### Basic Format
```
Title: [Slide Title]
Body: [Content]
```

### Examples

#### Bullet Points
```
Title: Key Findings
Body:
- First point
- Second point
- Third point
```

#### Numbered Lists
```
Title: Implementation Steps
Body:
1. First step
2. Second step
3. Third step
```

#### Mixed Format
```
Title: Marketing Strategy
Body:
Our approach includes:
- Market analysis
- Competitive positioning
- Implementation timeline:
  1. Phase one
  2. Phase two
```

#### Multiple Slides
For content that requires multiple slides, use the slide delimiter:
```
===SLIDE 1===
Title: Introduction
Body: Overview content here

===SLIDE 2===
Title: Details
Body: Detailed content here
```

## Google Docs Format Templates

Docs format templates use Markdown-style formatting to structure documents with headings, lists, and formatting.

### Basic Format
```
# Main Title

## Section Heading
Content paragraph

### Subsection
- Bullet point
- Another point

## Another Section
1. Numbered item
2. Another item
```

### Examples

#### Document with Multiple Sections
```
# Marketing Attribution Guide

## What is Attribution?
Attribution is the process of identifying which marketing efforts contribute to conversions.

## Common Models
- First-touch attribution
- Last-touch attribution
- Multi-touch attribution

## Implementation Steps
1. Define goals
2. Select appropriate tools
3. Analyze results
```

## Tips for Effective Templates

1. **Be specific**: Clearly indicate the structure you want
2. **Use appropriate formatting**: Choose bullet points for unordered lists, numbers for sequences
3. **Keep it concise**: Format templates should be clear and easy to understand
4. **Consider content length**: For slides, limit content to what fits on a single slide
5. **Use headings hierarchy**: For docs, use heading levels (# for main title, ## for sections, ### for subsections)

## How Templates Are Used

The RAG system will:
1. Take your question
2. Generate content from the knowledge base
3. Format that content according to your template
4. Create the document with the formatted content