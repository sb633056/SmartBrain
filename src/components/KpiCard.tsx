// src/components/KpiCard.tsx
import React from "react";


type Props = {
title: string;
value: string | number;
delta?: string;
children?: React.ReactNode;
};


export default function KpiCard({ title, value, delta, children }: Props) {
return (
<div className="border rounded-2xl p-4 shadow-sm bg-white">
<div className="flex items-start justify-between">
<div>
<div className="text-sm text-gray-500">{title}</div>
<div className="text-2xl font-semibold">{value}</div>
</div>
{delta && (
<div className="text-sm text-green-600 font-medium">{delta}</div>
)}
</div>
{children}
</div>
);
}