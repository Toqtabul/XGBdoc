export default function CodeBlock({ language = 'python', children }) {
  return (
    <div className="relative my-4">
      <div className="absolute top-0 right-0 px-3 py-1 text-xs text-gray-400 bg-[#1e272c] rounded-bl-lg rounded-tr-md">
        {language}
      </div>
      <pre className="bg-[#263238] text-[#aabbc3] p-4 pt-8 rounded-lg overflow-x-auto text-sm leading-relaxed">
        <code>{children}</code>
      </pre>
    </div>
  )
}
