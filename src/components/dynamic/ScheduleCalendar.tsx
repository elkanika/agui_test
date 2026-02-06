import React from 'react';
import { Calendar, Clock } from 'lucide-react';

interface ScheduleCalendarProps {
  data: {
    events: {
      title: string;
      date: string;
      time?: string;
      description?: string;
    }[];
  };
}

export const ScheduleCalendar: React.FC<ScheduleCalendarProps> = ({ data }) => {
  if (!data || !data.events) return <div>Invalid Calendar Data</div>;

  return (
    <div className="space-y-3 my-4">
      {data.events.map((event, idx) => (
        <div
          key={idx}
          className="flex bg-white border border-gray-200 rounded-lg p-3 shadow-sm hover:shadow-md transition-shadow"
        >
          <div className="flex-shrink-0 bg-blue-50 text-blue-600 rounded-lg p-2 flex flex-col items-center justify-center w-16 h-16 mr-4">
            <Calendar className="w-5 h-5 mb-1" />
            <span className="text-xs font-bold text-center leading-tight">
              {event.date}
            </span>
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-gray-900">{event.title}</h4>
            {event.time && (
              <div className="flex items-center text-sm text-gray-500 mt-1">
                <Clock className="w-3 h-3 mr-1" />
                <span>{event.time}</span>
              </div>
            )}
            {event.description && (
              <p className="text-sm text-gray-600 mt-1">{event.description}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
