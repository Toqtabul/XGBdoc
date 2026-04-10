export default function ParamTable({ items }) {
  return (
    <div className="overflow-x-auto my-4">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50">
            <th className="text-left p-3 font-semibold">Parameter</th>
            <th className="text-left p-3 font-semibold">Type</th>
            <th className="text-left p-3 font-semibold">Default</th>
            <th className="text-left p-3 font-semibold">Description</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => (
            <tr key={idx} className="border-b border-gray-200 hover:bg-gray-50">
              <td className="p-3">
                <code className="text-[#d63384] bg-gray-100 px-1.5 py-0.5 rounded text-xs">
                  {item.name}
                </code>
                {item.alias && (
                  <span className="text-gray-500 text-xs block mt-1">
                    alias: {item.alias}
                  </span>
                )}
              </td>
              <td className="p-3 text-gray-600">{item.type}</td>
              <td className="p-3">
                <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">
                  {item.default}
                </code>
              </td>
              <td className="p-3 text-gray-700">
                {item.description}
                {item.range && (
                  <span className="block text-xs text-gray-500 mt-1">
                    Range: {item.range}
                  </span>
                )}
                {item.options && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {item.options.map((opt, i) => (
                      <span key={i} className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                        {opt}
                      </span>
                    ))}
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
