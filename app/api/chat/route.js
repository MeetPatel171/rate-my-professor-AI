import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = `
You are an AI assistant designed to help students find professors that match their academic interests and needs. Your role is to:

    1. Analyze student queries about professor recommendations.
    2. Consider factors like academic field, research interests, teaching style, and course offerings.
    3. Provide the top 3 most relevant professor recommendations for each query.
    4. Use RAG (Retrieval-Augmented Generation) to access and incorporate up-to-date information about professors from your knowledge base.
    5. Offer brief explanations for why each recommended professor might be a good fit.
    6. Maintain a friendly and helpful tone while interacting with students.
    7. Respect privacy and only share publicly available information about professors.

Respond to each student in a nice formatted output query with well-reasoned recommendations to help them find the most suitable professors for their needs.
`

export async function POST(req) {
    const data = await req.json()
    // We'll add more code here in the following steps

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      })
      const index = pc.index('rag').namespace('ns1')
      const openai = new OpenAI()

      const text = data[data.length - 1].content
      const embedding = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: text,
      encoding_format: 'float',
      })

      const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
      })

      let resultString = ''
    results.matches.forEach((match) => {
    resultString += `
    Returned Results:
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await openai.chat.completions.create({
        messages: [
        {role: 'system', content: systemPrompt},
        ...lastDataWithoutLastMessage,
        {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-3.5-turbo',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
        const encoder = new TextEncoder()
        try {
            for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content
            if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
            }
            }
        } catch (err) {
            controller.error(err)
        } finally {
            controller.close()
        }
        },
    })
  return new NextResponse(stream)
  }